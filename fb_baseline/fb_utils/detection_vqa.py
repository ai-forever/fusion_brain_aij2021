import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from scipy.optimize import linear_sum_assignment
import numpy as np
from torchvision.ops import generalized_box_iou

## Model ##
class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Resnet50Backbone(nn.Module):
    
    def __init__(self, pretrained=True):
        super().__init__()
        self.num_channels = 512
        self.backbone = resnet18(pretrained=pretrained)
        
    def forward(self, img):
        x = self.backbone.conv1(img)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        return x


class FeedForwardComponent(nn.Module):

    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


class CrossAttentionLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, pf_dim, dropout=0.0):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

        self.self_attention = nn.MultiheadAttention(hid_dim, n_heads, dropout)
        self.feed_forward = FeedForwardComponent(hid_dim, pf_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, img, text, text_mask=None):
        img = img.transpose(0, 1)
        text = text.transpose(0, 1)
        
        _img, _img_attention = self.self_attention(img, text, text, key_padding_mask=text_mask)
        img = self.self_attn_layer_norm(img + self.dropout(_img))

        _img = self.feed_forward(img)
        img = self.ff_layer_norm(img + self.dropout(_img))

        return img.transpose(0, 1), _img_attention
############

## Evaluation part ##
def vqa_evaluation(model, images, tokens, attention_masks, max_answer_length):
    back_out = model.backbone(images)
    patchs = model.input_proj(back_out).flatten(-2).transpose(-1, -2)
    gpt_img = model.gpt_model(inputs_embeds=patchs).last_hidden_state
    
    answer_logits = []
    bs = gpt_img.shape[0]
    for _ in range(max_answer_length):
        gpt_text = model.gpt_model(input_ids=tokens, attention_mask=attention_masks).last_hidden_state
        for layer in model.cross_attention:
            gpt_text, _ = layer(gpt_text, gpt_img)

        logits = model.tokens_embed(gpt_text)
        last_logits = logits[:, -1, :]
        answer_logits.append(last_logits)

        new_tokens = torch.multinomial(last_logits.softmax(-1), num_samples=1)
        tokens = torch.cat([tokens, new_tokens], dim=-1)
        new_attention_masks = torch.ones(bs, 1).to(attention_masks.device, dtype=attention_masks.dtype)
        attention_masks = torch.cat([attention_masks, new_attention_masks], dim=1)

    return torch.stack(answer_logits, dim=1)

def detection_evaluation(model, images, input_ids, attention_masks, cor_treshhold, treshhold):
    back_out = model.backbone(images)
    patchs = model.input_proj(back_out).flatten(-2).transpose(-1, -2)
    gpt_img = model.gpt_model(inputs_embeds=patchs).last_hidden_state
    norm_gpt_img = F.normalize(gpt_img, p=2, dim=-1)
    
    boxes = []
    for tokens, attention_mask in zip(input_ids, attention_masks):
        gpt_text = model.gpt_model(input_ids=tokens, attention_mask=attention_mask).last_hidden_state
        norm_gpt_text = F.normalize(gpt_text, p=2, dim=-1)
        corr_matrix = torch.matmul(norm_gpt_img, norm_gpt_text.transpose(-1, -2))
        cut_gpt_img = gpt_img[corr_matrix.mean(-1) > cor_treshhold].unsqueeze(0)
        if cut_gpt_img[1] == 0:
            boxes.append(torch.tensor([]).to(cut_gpt_img.device))
            continue
        text_mask = attention_mask.type(torch.bool)
        for layer in model.cross_attention:
            cut_gpt_img, _ = layer(cut_gpt_img, gpt_text, ~text_mask)
        cut_gpt_img = model.detection_pool(cut_gpt_img)
        
        output_logits = model.bbox_embed(cut_gpt_img).sigmoid()
        output_boxes = output_logits[output_logits[:, :, -1] > treshhold][:, :-1]
        boxes.append(output_boxes)
    
    return boxes
############


## Loss ##
def box_xywh_to_xyxy(x):
    x, y, w, h = x.unbind(-1)
    b = [(x), (y), (x + w), (y + h)]
    
    return torch.stack(b, dim=-1)


class DetectionCriterion(nn.Module):
    
    def __init__(self, losses, temperature):
        super().__init__()
        self.losses = losses
        self.temperature = temperature
    
    def _get_idx(self, pred_boxes, targets):
        boxes_idx = list(map(
            lambda x: torch.argmax(generalized_box_iou(box_xywh_to_xyxy(x[0]), box_xywh_to_xyxy(x[1][:, :-1])), -1),
            zip(targets, pred_boxes)
        ))
        
        return boxes_idx
        
    def loss_classification(self, outputs, targets, boxes_idx, num_boxes):
        pred_probs = [t[:, -1] for t in outputs["pred_logits"]]
        target_labels = [torch.zeros_like(pred_prob) for pred_prob in pred_probs]
        for i, idx in enumerate(boxes_idx):
            target_labels[i][idx] = 1.
        
        pred_probs = torch.cat(pred_probs)
        target_labels = torch.cat(target_labels)
        loss = F.binary_cross_entropy(pred_probs, target_labels)
        
        return {"loss_classification": loss}
    
    def loss_contrastive(self, outputs, targets, boxes_idx, num_boxes):
        norm_img_emb = outputs['proj_queries']
        norm_tokens_emb = outputs['proj_tokens']
        corr_tensor = torch.tensordot(norm_img_emb, norm_tokens_emb.permute(2, 0, 1), dims=1).transpose(1, 2)
        corr_matrix = corr_tensor.mean((-1, -2)) * np.exp(self.temperature)
        
        bs = corr_matrix.shape[0]
        labels = torch.arange(bs).to(corr_matrix.device)
        loss_tok2obj = F.cross_entropy(corr_matrix, labels)
        loss_obj2tok = F.cross_entropy(corr_matrix.transpose(0, 1), labels)
        
        contrastive_loss = (loss_tok2obj + loss_obj2tok) / 2.

        return {
            "loss_contrastive": contrastive_loss
        }
    
    def loss_boxes(self, outputs, targets, boxes_idx, num_boxes):
        src_boxes = torch.cat([t[idx][:, :-1] for t, idx in zip(outputs["pred_logits"], boxes_idx)], dim=0)
        target_boxes = torch.cat([t for t in targets], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1. - torch.diag(
            generalized_box_iou(
                box_xywh_to_xyxy(src_boxes), box_xywh_to_xyxy(target_boxes)
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        
        return losses

    def get_loss(self, loss, outputs, targets, boxes_idx, num_boxes, **kwargs):
        loss_map = {
            "classification": self.loss_classification,
            "contrastive": self.loss_contrastive,
            "boxes": self.loss_boxes
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        
        return loss_map[loss](outputs, targets, boxes_idx, num_boxes, **kwargs)

    
    def forward(self, outputs, targets):
        boxes_idx = self._get_idx(outputs["pred_logits"], targets)
        num_boxes = float(sum([boxes.shape[0] for boxes in targets]))

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, boxes_idx, num_boxes))
            
        return losses
############

