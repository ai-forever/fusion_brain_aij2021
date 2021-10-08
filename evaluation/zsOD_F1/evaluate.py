import json
import argparse
import os
import numpy as np
import itertools


def xywh_to_xyxy(bbox):
    '''
    [xmin,ymin,w,h] -> [xmin,ymin,xmax,ymax]
    '''
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def iou(box1, box2):
    '''
    IoU between 2 bboxes, each box is [xmin,ymin,xmax,ymax].

    Args:
      box1: (list) bbox, sized (4,).
      box2: (list) bbox, sized (4,).
    Return:
      float: iou
    '''
    lt = np.zeros(2)
    rb = np.zeros(2)  # get inter-area left_top/right_bottom
    for i in range(2):
        if box1[i] > box2[i]:
            lt[i] = box1[i]
        else:
            lt[i] = box2[i]
        if box1[i + 2] < box2[i + 2]:
            rb[i] = box1[i + 2]
        else:
            rb[i] = box2[i + 2]
    wh = rb - lt
    wh[wh < 0] = 0  # if no overlapping
    inter = wh[0] * wh[1]
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter / (area1 + area2 - inter)
    if np.isnan(iou):
        iou = 0
    return iou


def evaluate(true_annot, predictions):
    '''
    Args:
      ref_path: (string) path to pred.json
      pred_path: (string) path to true.json
    Return:
      float: f1_score
    '''


    assert set(true_annot.keys()) == set(predictions.keys()), 'Not all images have predictions!'

    fp = 0
    fn = 0
    tp = 0

    for img in true_annot:
        for label in true_annot[img]:
            true = true_annot[img][label]
            assert label in predictions[
                img], f'There are no prediction for label "{label}" for image {img} in requests!'

            pred = predictions[img][label]

            if pred == [[]]:
                pred = []

            if len(pred) == 0 and len(true) == 0:
                continue
            elif len(pred) > 0 and len(true) == 0:
                fp += len(pred)
            elif len(pred) == 0 and len(true) > 0:
                fn += len(true)

            else:

                pairs = list(itertools.product(true, pred))
                pairs_iou = [(el[0], el[1], iou(xywh_to_xyxy(el[0]), xywh_to_xyxy(el[1]))) for el in pairs]
                for _, group in itertools.groupby(pairs_iou, key=lambda x: x[0]):  # true
                    if np.all(np.array([i for _, _, i in group]) < 0.5):
                        fn += 1

                for _, group in itertools.groupby(sorted(pairs_iou, key=lambda x: x[1]), key=lambda x: x[1]):  # pred
                    if np.all(np.array([i for _, _, i in group]) < 0.5):
                        fp += 1
                    else:
                        tp += 1

    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if precision > 0 or recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0
    return f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path', type=str, default='./true_zsOD.json', help="the path to the gold annotations")
    parser.add_argument('--pred_path', type=str, default='./prediction_zsOD.json',
                        help="the path to the file with predictions")

    args = parser.parse_args()

    with open(args.ref_path) as f:
        true_annot = json.load(f)

    with open(args.pred_path) as f:
        predictions = json.load(f)

    f1 = evaluate(true_annot, predictions)

    print(f1)
