# Articles on multi-tasking and multimodal models

The development of multitasking and multimodal models is now an active field of research that saves computational resources and training time. It is certainly possible, lead to the creation of strong artificial intelligence. 

1. [UniT: Multimodal Multitask Learning with a Unified Transformer](https://arxiv.org/pdf/2102.10772.pdf) (Unified Transformer (UniT), ```Hu, Singh, 2021```), which is part of the framework [MMF](https://github.com/facebookresearch/mmf)  for the construction of multimodal models. In the experiments, UniT simultaneously handles 7 tasks(in text, visual, and shared text-visual domains) on 8 datasets (MS-COCO, VG, VQAv2, SNLI-VE, QNLI, MNLI-mm, QQQP, SST-2). The model is based on the
transformer encoder-decoder architecture: for each data type (modality, in this case - text and visual), authors use its encoder; make
predictions on each task with shared decoder over the encoded input representations. Decoders output is followed by task-specific output heads (for all tasks except object detection, it is a two-layer perceptron) to make the final prediction:

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/unit.png" width="60%">
</p>

At each iteration during training, authors randomly select a task and a dataset to fill a batch of samples;  for each task prior, specify a sampling probability.

2. [Attention Bottlenecks for Multimodal Fusion](https://arxiv.org/abs/2107.00135) (Multimodal Bottleneck Transformer (MBT), `Nagrani, Arsha et al., 2021`) - an original approach to deal with multi-modal inputs (video and audio, mostly) with middle-level fusion of it. 

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/mid_fuse.png" width="60%">
</p>

Almost all the layers modalities are processed separately by a transformer, and only near the top (2 - 4 layers) the fusion is done: 
    - StepA: neurons modality<sub>1</sub> are concatenated with a small number B of so called multimodal bottlenecks (authors used B = 4) and then the whole self-attention processing (modality<sub>1</sub> + multimodal bottlenecks) is done, and then
    - StepB: neurons of modality<sub>2</sub> are concatenated with these B multimodal bottlenecks (corresponding output of StepA) and then the whole self-attention processing (modality<sub>2</sub> + multimodal bottlenecks) is done.
    
<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/mbt.png" width="60%">
</p>

Main interesting part here is to share multi-modal information through a very tight information bottleneck at the same time without adding computational complexity.  
For transformer authors used ViT-B architecture. 

3. [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206) and [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795) (`Jaegle, Andrew et al., 2021`) - the novel method of dealing with attentions for multi-modal data with linear complexity on either input size or output size. Two core ideas:
    - Iterative attention, where the same input can be fed into different depth - like RNN procedure (authors claims that this idea lies in how human brain works) - where the parameters of transformers can be shared, and

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/perceiver.png" width="60%">
</p>

    - Cross-attention, where either query is latent flow and key-values are input data (for feeding data into the model), or query is the output structure (e.g., the list of positions of pixels and task<sub>id</sub>)

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/perceiverIO.png" width="60%">
</p>

Important note here is how to feed the information (either input or output). Authors propose different schemes of positional encoding (includig Fourier Features), or even the learned encoding (so there is no need in explicit tokenization). 

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/perceiverIO_emb.png" width="60%">
</p>

Latent transformer architecture is GPT-2. 

4. [Multi-Task Deep Neural Network](https://github.com/namisan/mt-dnn) (MT-DNN, ```Liu, He et al., 2019```) –  a single model was created to address multiple natural language understanding (NLU) tasks. The the lower layers are shared across
all tasks, while the top layers are task-specific. Authors use a multilayer bidirectional Transformer encoder. Unlike the BERT
model that learns the representation via pre-training, MT-DNN learns the representation using multi-task objectives, in addition to pre-training.

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/mt-dnn.png" width="60%">
</p>

The training procedure of MT-DNN consists of two stages: pretraining and multi-task learning. In the multi-task learning stage, a minibatch is selected in each epoch (e.g., among all 9 GLUE tasks)-the model's weights are updated according to the task-specific objective for each task. This approach approximately optimizes the sum of all multi-task objectives. The authors state that multi-task learning benefits from a regularization (alleviating overfitting  to a specific task) effect that leads to more general representations. MT-DNN is learned on three popular NLU benchmarks: SNLI, GLUE, and SciTail, solves four different types of tasks such as single-sentence classification, pairwise text classification, text similarity, and relevance ranking. 

5. [OmniNet: A unified architecture for multi-modal multi-task learning](https://arxiv.org/pdf/1907.07804.pdf) (`Pramanik et al., 2020`) -model [OmniNet](https://github.com/subho406/OmniNet) consists of two sub-networks, called peripheral networks: one is used
to encode images and video, other is used to encode text.  As an image and video encoder is used the pre-trained convolutional neural network ResNet-152. For text encoding, authors use pre-trained subword embeddings obtained using BPE. Coded data concatenated with data type embedding and entered in Central Neural Processor. Data with a temporal dimension pass through the encoder block; data without a temporal dimension is then reshaped. All received matrixes are saved in a cache that enters the decoder block:

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/omninet.png" width="70%">
</p>

There are tasks embeddings, several sets of output embeddings for different output data types, and separate classifiers for different tasks. Multitasking is achieved by using the HogWild training approach: in addition to a global copy of the model, creating local copy for each task. The locally computed gradients are copied to the global model asynchronously to update the weights of the global model.
Model learning unseen tasks such as video captioning and video question answering. 


6. [12-in-1: Multi-Task Vision and Language Representation Learning](https://arxiv.org/pdf/1912.02315.pdf) (`Lu, Goswami et al., 2020`) - the article describes a model based on architecture [ViLBERT](https://arxiv.org/pdf/1908.02265.pdf) (`Lu et al., 2019`), in which image and text input collaborate by co-attention:

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/vilbert.png" width="80%">
</p>

Image and text to which the corresponding token task is added, encoded by two blocks with similar architecture to BERT.
These modules are pretrained to the masked element prediction tasks and predicting correlation between the input data.
Authors use 12 datasets from four broad categories of tasks, including visual question answering, caption-based image retrieval,
grounding referring expressions, and multi-modal verification. 6 task heads are trained to perform these tasks. Authors introduce the dynamic stop and go (DSG) mechanism, which allows stop using small datasets on more iterations and continues to use large datasets.


7. [M3P: Learning Universal Representations via Multitask Multilingual Multimodal Pre-training](https://arxiv.org/pdf/2006.02635.pdf) (`Ni et al., 2021`) —  model that combines multilingual pre-training and multimodal pre-training g into a unified framework via
multitask pre-training:

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/m3p.png" width="80%">
</p>

Multitask training is employed in the pre-training stage to optimize all pre-training objectives simultaneously for better performance. First, the model is pretrained on the masked token prediction task. The model uses three types of data streams:  one language text+ image;  text with language change, Multimodal Code-switched Training (MCT); a combination of the first two cases. Experiments are performed on the multilingual image retrieval task across two benchmark datasets, including  Multi30K - extended Flickr30K ( English, German, French, and Czech captions) and MSCOCO (English, Chinese and Japanese  captions).

8. [HyperGrid Transformers: Towards A Single Model for Multiple Tasks](https://openreview.net/pdf?id=hiq1rHO8pNT) – the authors propose the HyperGrid Transformer for the efficient modeling of multiple tasks. This paper proposes a decomposable hypernetwork (the network that generates weights for the base model) that learns grid-wise projections that help to specialize regions in weight matrices for different tasks. In order to construct the proposed hypernetwork,the method learns the interactions and composition between a global (task-agnostic) state and a local task-specific state. Local and Global vectors compose to form a gating matrix which is expanded (by repeat) up to the size of the Transformer matrix to construct task-adaptive weight matrices:

<p align="center">
  <img src="https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/hypergrid.png" width="80%">
</p>

The authors conduct experiments on benchmarks GLUE and SuperGLUE (NLP and NLU tasks), using the base T5 model with HyperGrid layers. Results show a single model with performance (Avg 85.0 on GLUE and 73.6 on SuperGLUE), not far below the quality finetuned T5 model (Avg 85.7 on GLUE and 74.8 on SuperGLUE), while being 16 times more parameter efficient. 
 
