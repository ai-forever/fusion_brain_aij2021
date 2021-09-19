# Fusion Brain Challenge

The current task suggests developing a single multitask model, which would successfully complete such multimodality subtasks as **Code2code translation (С2С), Zero-shot object detection (OD), Handwritten Text Recognition (HTR), Visual Question Answering (VQA)** and be able to break an integral metric of baseline proposed by the Arranger, as well as metrics for each subtask.

We provide [a concept of a single model](https://colab.research.google.com/drive/1YAkxWG0dRKPtqy9CZxFPvCNCCXvMGr65?usp=sharing%22%20target%3D%22_parent%22%3E%3Cimg%20src%3D%22https%3A%2F%2Fcolab.research.google.com%2Fassets%2Fcolab-badge.svg%22%20alt%3D%22Open%20In%20Colab) that is trained on several tasks related to different modalities (visual, audio and text). The concept is inspired by an article ["Pretrained Transformers as Universal Computations Engines"](https://arxiv.org/pdf/2103.05247.pdf) (```Lu et al., 2021```) that examines the ability of pretrained language models based on the Transformer architecture to form qualitative representations of arbitrary data sequences – thus, generalizing to other modalities with minimal finetuning. The basis of the architecture proposed in the concept is the pretrained GPT-2 language model; experiments are carried out both with a "frozen" model (Frozen Pretrained Transformer), and with a model in which all layers are trained on three modalities simultaneously.

Uploading solutions to the Competition platform will become available from **04/10/2021**.

## General solution format

Participants should create a zip archive with a trained model and a set of scripts for model prediction. The participant shall upload this archive to the competition platform. Then, the archive shall be unzipped to a docker container. Solutions may be run with the use of any image available for downloading from DockerHub. If necessary, you can prepare your own image, add necessary software and libraries to it (see the manual on creating Docker images); to use it, you will need to publish it on DockerHub.

The archive root must contain the metadata.json file containing the following:
```
{
    "image": "sberbank/fb-python",
    "entry_point": "python run.py $PATH_INPUT $PATH_OUTPUT"
}
```
Where ```image``` is a field with the docker image name, in which the solution will be run, ```entry_point``` is a command that runs the solution. For solution, the archive root will be the current directory. During the run, the ```DATASETS_PATH``` environment variable shall contain the route to relevant open datasets accessible from the container with the solution.

An argument accepted by the script for model inference should be represented by a path to the folder with the content to be used for prediction. Let us assume that the argument shall be represented by the ```fusion_brain``` folder. Names of ```fusion_brain``` subfolders shall correspond to names of subtasks to be completed by the single model. Each subfolder (HTR, OD, VQA, C2C) shall have the content needed for predictions.

The data structure is as follows:

* fusion_brain
  * HTR
    * images 
  * OD
    * images
    * requests.json
  * VQA
    * images
    * questions.json
  * C2C 
    * requests.json

The single model should generate predictions for each subtask in the ```prediction_{TASK_NAME}.json``` format, i.e. there should be four files after the model inference: ```prediction_HTR.json, prediction_OD.json, prediction_VQA.json, prediction_C2C.json```. These files should be located in the root of uploaded file with the solution.

After that, correct answers in the ```true_{TASK_NAME}.json``` format shall be added to the container and a script for calculation of metrics for each subtask shall be run. The final metric shall be calculated as a sum of metrics for each subtask (see below).

# Subtask 1 - Code2code translation

## Description

A task of translation from one programming language into another is a standard part of the wide-ranging repertoire of ML4Code. As of today, there are several alternate solutions – in line with both supervised learning, where a training dataset is represented by a parallel corpus (baseline model of the CodeXGLUE benchmark with CodeBERT as a coder in the coder-decoder-type architecture, ```Lu et al., 2021```), and unsupervised one, including pretraining of cross-lingual language model on monolingual corpora (TransCoder, ```Lachaux et al., 2020```). 

Cases where a source language and a target language have different type systems are particularly problematic. It is that very class that our subtask belongs to: we should translate from a statically-typed language (Java) to a dynamically-typed one (Python). The input is a Java function and the output should be the similar function in Python.

## Data

**Train.** It is suggested that the training dataset should be represented by train (5,937 programmer tasks) and val (845) parts of the [AVATAR](https://github.com/wasiahmad/AVATAR) parallel corpus consisting of pairs of similar functions/programs, one of which is written in Java, while the other is written in Python. Due to the fact that the dataset contains 1-25 candidate solutions in both programming languages, it shall be possible to generate 1-25 parallel examples for each task. The dataset authors suggest choosing no more than three solutions for each language - therefore, there will be no more than nine training examples for one problem. We suggest using a [dataset](https://dsworks.s3pd01.sbercloud.ru/aij2021/%D0%A12%D0%A1_translation/CodeToCodeTranslation_train_java_python.jsonl) generated in this very manner.

The file have the jsonl format with java and python fields:
```
{"java":"import java . util . Scanner ; \u00a0 public class A1437 { \u00a0 public static void main ( String [ ] args ) { Scanner in = new Scanner ( System . in ) ; int T = in . nextInt ( ) ; for ( int t = 0 ; t < T ; t ++ ) { int L = in . nextInt ( ) ; int R = in . nextInt ( ) ; boolean possible = R < 2 * L ; System . out . println ( possible ? \" YES \" : \" NO \" ) ; } } \u00a0 }\n","python":"t = int ( input ( ) ) NEW_LINE ans = [ ] NEW_LINE for i in range ( t ) : l , r = [ int ( x ) for x in input ( ) . split ( ) ] NEW_LINE if ( 2 * l ) > r : NEW_LINE INDENT ans . append ( \" YES \" ) else : NEW_LINE ans . append ( \" NO \" ) NEW_LINE DEDENT for j in ans : print ( j ) NEW_LINE\n"}
```

To create a parallel training corpus, you can also use [CodeNet](https://github.com/IBM/Project_CodeNet), which contains solutions in 4 languages (C++, C, Python and Java) to 4,000 programming problems, extracted from two online judge web sites: AtCoder (the AVATAR dataset also includes solutions from this resource for some tasks) and AIZU Online Judge. However, it should be taken into account that the solutions of one programming problem in different languages are, at least, type IV clones (preserving source code semantics, but having considerable differences in syntax), but they are not guaranteed to be identical to each other, adjusted for peculiarities of languages (literal translation).

**Test public.** The public leaderboard shall be generated according to the results of model prediction check based on a test set (1,693) from the AVATAR dataset.

**Test private.** The private test dataset is hidden from participants. Its format is similar to the public test set.

## Quality metric

**CodeBLEU** is a metric suggested in ```Ren et al., 2020``` to adjust the BLEU metric, which is normally used to assess the translation quality for natural languages, to the source code translation assessment; such adjustment supplementing the n-grams comparison by the comparison of corresponding abstract syntax trees in the reference code and translated code (thus we consider the code syntax), as well as juxtaposing the data flows (thus we consider the code semantics).

CodeBLEU is a weighted combination of four components:

![image](https://latex.codecogs.com/svg.image?\textrm{CodeBLEU}&space;=&space;\alpha&space;\cdot&space;\textrm{BLEU}&space;&plus;&space;\beta&space;\cdot&space;\textrm{BLEU}_{weight}&space;&plus;&space;\gamma&space;\cdot&space;\textrm{Match}_{ast}&space;&plus;&space;\delta&space;\cdot&space;\textrm{Match}_{df},)

where BLEU is the standard BLEU metric ```Papineni et al., 2002```, BLEU<sub>weight</sub> is the weighted comparison of n-grams (with the tokens differing by their significance and the match of certain tokens for the translated and ‘golden’ functions is assigned more weight), Match<sub>ast</sub> is the comparison metric for abstract syntax trees of the translated code and of the reference code, and Match<sub>df</sub> reflects the similarity of data flows produced by hypotheses and correct functions.

Let us discuss each component of the metric in more detail.

* BLEU is based on calculation of n-grams found in the translation and the reference sequence calculated as follows:

![image](https://latex.codecogs.com/svg.image?\textrm{BLEU}&space;=&space;\textrm{BP}&space;\cdot&space;\textrm{exp}&space;(\sum_{n=1}^{N}&space;w_{n}&space;\log&space;p_{n}),)

where BP is a penalty for too short translations, which shall be calculated as the number of tokens in the model-suggested translation divided by the number of tokens in the reference sequence; and the second part of the formula is a geometric mean of a modified n-gram precision:

![image](https://latex.codecogs.com/svg.image?\textrm{p}_{n}&space;=&space;\frac{\sum_{C&space;\in&space;Candidates}&space;\sum_{n\text{-}gram&space;\in&space;C}&space;Count_{clip}&space;\textrm{(n-gram)}}{\sum_{C\textquotesingle&space;\in&space;Candidates}&space;\sum_{n\text{-}gram\textquotesingle&space;\in&space;C\textquotesingle}&space;Count&space;\textrm{(n-gram\textquotesingle)}})

for n-grams with the length from 1 to N, multiplied by corresponding positive weights w<sub>n</sub> that in total make 1.

* As opposed to the standard BLEU metric, the BLEU<sub>weight</sub> metric calculates the accuracy of n-grams coincidence by using the weight factor (![image](https://latex.codecogs.com/svg.image?\mu_{n}^{i})) with its value higher for keywords in a programming language than for any other tokens:

![image](https://latex.codecogs.com/svg.image?\textrm{p}_{n}&space;=&space;\frac{\sum_{C&space;\in&space;Candidates}&space;\sum_{i=1}^{l}&space;\mu_{n}^{i}&space;Count_{clip}&space;(C(i,&space;i&plus;n))}{\sum_{C\textquotesingle&space;\in&space;Candidates}&space;\sum_{i=1}^{l}&space;\mu_{n}^{i}&space;Count&space;(C\textquotesingle(i,&space;i&plus;n))},)

where C(i, i+n) is an n-gram starting in the i place and ending in the i+n place, Count<sub>clip</sub> has the same meaning as that in the standard BLEU metric, it is the maximum number of n-grams found in both translated code and reference set. The keywords list is pre-determined for each programming language.

* For a source code, its syntax structure could be expressed as an abstract syntax tree (AST), thus enabling to compare the translated and reference functions on the level of subtrees generated by an AST parser. Since we are interested in the syntax, the AST leaves containing variables could be omitted. Match<sub>ast</sub> shall be calculated according to the following formula:

![image](https://latex.codecogs.com/svg.image?\textrm{Match}_{ast}&space;=&space;\frac{\textrm{Count}_{clip}(\textrm{T}_{cand})}{\textrm{Count}(\textrm{T}_{ref})},)

where Count(T<sub>ref</sub>) is a total number of subtrees in the reference code, Count<sub>clip</sub>(T<sub>cand</sub>) is a number of subtrees in the translated code that have matched the subtrees of reference functions. This metric allows assessing the quality of a code translated in terms of its syntax.

* The translated code and reference code are also compared by their semantics, using data flows (```Guo et al., 2020```) when the source code is represented as a graph with its nodes being variables and its edges representing the ‘genetic’ relations between the nodes (denoting where the value of each variable comes from). The Match<sub>df</sub> metric shall be calculated according to the following formula:

![image](https://latex.codecogs.com/svg.image?\textrm{Match}_{df}&space;=&space;\frac{\textrm{Count}_{clip}(\textrm{DF}_{cand})}{\textrm{Count}(\textrm{DF}_{ref})},)

where Count(DF<sub>ref</sub>) is a total number of data flows in the reference code; Count<sub>clip</sub>(DF<sub>cand</sub>) is a number of data flows in the translated code that have matched the reference code.

## Solution format

Participants should create an archive with a trained model and a set of scripts for model prediction. The participant shall upload this archive to the competition platform. Then, the archive shall be unzipped to a docker container, while the system shall add the data for prediction to the container space. Such data shall include:

* The ```requests.json``` file. It is a dictionary in the following format: ```{ "0": "import java . util . Scanner ; ..." , ... }```. Keys shall be represented by sample indices, while values shall be represented by lines of functions/programs in Java that should be translated into Python.

The participant’s model should translate all examples from the requests.json file and generate the ```prediction_С2С.json``` file. It is a dictionary in the following format: ```{ "0": "def find ( x , par ) : NEW_LINE INDENT if par [ x ] == x : ..." , ... }```. Keys shall be represented by sample indices, while values shall be represented by translations of functions/programs into Python. Please, pay attention to the fact that since Python uses indentations to identify logic blocks in codes, the line of translation into Python includes such special tokens as ```INDENT```, ```DEDENT```. 

After inference, the metric calculation script shall compare the ```prediction_С2С.json``` and ```true_С2С.json``` files, and then display the final value of the CodeBLEU metric.

# Subtask 2 - Zero-shot object detection

## Description

* It is necessary to determine the class of an object shown in a photo (or classes, if there are several objects). For example, there can be such entities/objects on the photo as “human“, “car“, “apple“.

* At the same time, it is necessary to determine the location and scale of each object shown in a photo. The object location shall be described with the so-called bounding box (bbox). This is a rectangle to be drawn most accurately around the object. The rectangle position shall be set with four numbers – X, Y, W, H, where:

    * X is a horizontal coordinate of the top left corner
    * Y is a vertical coordinate of the top left corner
    * W is the rectangle width
    * H is the rectangle height

For each object in a photo, the model predictions should be represented by bbox coordinates and a class tag. An example of the result of object detection model work is shown in the next picture:

![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/od.png)

Within the framework of our competition, the task is defines as zero-shot object detection. Zero-shot in the task description means that the model should be able to succesfully make predictions on a dataset completely differing from the training one. A standard object detection model is expected to predict one class out of a limited set of classes determined during the model training. A zero-shot model is expected to detect classes not found in the training set.

The set of possible classes for each image is fed into a model as a query. A query may contain classes in both Russian and English.

At the prediction stage, the model input shall contain two entities: an image and a natural-language query. The query is formatted as a text line containing a list of labels to search from. Example: “dog, bicycle, car, cake, airplane”. The query contains both correct tags (objects actually present in the picture) and some incorrect ones. Their combination makes a single search space for the model. The model should output a list of predicted tags with the corresponding bounding box coordinates.

## Data

**Train.** It is suggested that training should be based on a popular dataset called MS-COCO.

[Images](http://images.cocodataset.org/zips/train2017.zip)  
[Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

**Test public.** The public test dataset is generated from a part of the VisualGenome dataset; a set of classes in it is hidden from participants.

**Test private.** The private test dataset is hidden from participants, just as a set of classes in it.

## Quality metric

The quality will be evaluated using the **mean average precision** (mAP) metric. To calculate mAP, you should choose IoU (intersection over union) first. This metric evaluates the match quality of the predicted bbox and the reference one. It shall be calculated as the intersection area of these two bboxes divided by their combination area:

![image](https://latex.codecogs.com/svg.image?\textrm{IoU}&space;=&space;\frac{&space;\textrm{Intersection}}{&space;\textrm{Union}})

For each pair (prediction/true), IoU ranges from 0 to 1. The cutting threshold for IoU is 0.5, meaning that all bboxes with their IoU below 0.5 are deemed false predictions.

For each class with a certain IoU value, the Precision-Recall curve shall be drawn and the area below the curve shall be calculated. The calculations shall be averaged by class to arrive at the final mAP metric.

## Solution format

Participants should create an archive with a trained model and a set of scripts for model prediction. The participant shall upload this archive to the competition platform. Then, the archive shall be unzipped to a docker container, while the system shall add the data for prediction to the container space. Such data shall include:

* The ```images``` folder.  It is a set of images to make predictions for. It contains files in the following format: ```0.jpg, 1.jpg ...```.
* The ```requests.json``` file. It is a dictionary in the following format: ```{ "0.jpg": ["tree", "clock", "book"] , ... }```. Keys shall be represented by respective names of files from the images folder, while values shall be represented by a list of classes to be detected in the respective image (query). As we said before, the list of classes may be whether in Russian or in English. Therefore, the query contains the list of labels to search from (for example, “dog, bicycle, car, cake, airplane”). **The query contains both correct tags belonging to objects actually present in the image and some incorrect labels (there no respective objects in the picture)**.

The participant’s model should make predictions for all images from the images folder and generate a ```prediction_OD.json``` file. It is a dictionary in the following format: ```{"0.jpg": [["dog", 0.5, 473.07, 395.93, 38.65, 28.67], ["cat", 0.6, 0.0, 101.15, 452.3, 319.43]], ...```. Keys shall be represented by names of files from the images folder, while values shall be represented by model predictions for the respective images. These predictions should contain detected classes, together with coordinates of bounding boxes for the respective image: the key “name of file with the image” shall be used for showing a list of objects predicted on it. Format of one element in the list: ```["dog", 0.5, 473.07, 395.93, 38.65, 28.67]``` (six elements divided by commas). The first element is the class name, the second is the model score, and then we have four bbox coordinates in the ```xywh``` format. The nested list may contain the unlimited number of elements – these are all objects predicted by the model for the respective image.

Then, the system shall compare the file with predictions with the ```true_OD.json``` file containing correct answers, and display the final mAP metric.

# Subtask 3 - Handwritten Text Recognition

## Description

Participants are given the task to recognize a handwritten text in the picture. The model input is an image with handwritten text. The model output should be a text line corresponding to the image content (in this case - the line “последовал”):

![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/htr_posledoval.png)

## Data

**Train.** We provide a training [dataset](https://dsworks.s3pd01.sbercloud.ru/aij2021/htr/train.zip) consisting of two different datasets. The first one is a manually collected dataset of school copybooks. Images in this dataset are represented by individual words in a text written on a copybook page. The second part consists of a popular dataset called IAM. It is a set of handwritten words in English.

**Test public.** The public leaderboard shall also be calculated with regard to copybook datasets and IAM.

**Test private.** The private test dataset is hidden from participants. It is also a dataset for text recognition in a format similar to training dataset. However, we do not provide any information on dataset details.

## Quality metric

The key metric used to evaluate the participants’ solutions shall be represented by the formula: **1 - CER**, where CER is the character error rate metric. It shall be calculated as follows:

![image](https://latex.codecogs.com/svg.image?\text{CER}&space;=&space;\frac{\sum_{i=1}^n&space;\text{dist}_{c}(pred_i,&space;true_i)}{\sum_{i=1}^n&space;\text{len}_{c}(true_i)},)

where dist<sub>c</sub> is the Levenshtein distance calculated for tokens (including spaces), while len<sub>c</sub> is the line length in characters.

The **1 - CER** metric varies from 0 to 1, where 0 is the worst value and 1 is the best one.

## Solution format

Participants should create an archive with a trained model and a set of scripts for model prediction. The participant shall upload this archive to the competition platform. Then, the archive shall be unzipped to a docker container, while the system shall add the data for prediction to the container space. Such data shall include:

* The ```images``` folder.  It is a set of images to make predictions for. It contains files in the following format: ```0.jpg, 1.jpg ...```. Each file contains graphic images of characters to be translated into text characters (text lines).

The participant’s model should make predictions for all images from the images folder and generate a ```prediction_HTR.json``` file. It is a dictionary in the following format:  ```{"0.txt": "<predicted text in the picture>" , "1.txt": "<predicted text in the picture>" , ... }```. Keys shall be represented by respective names of files from the images folder, while values shall be represented by predicted lines in the respective images. If there is no prediction for a name.png file with the image, i.e. keys of the ```prediction_HTR.json``` dictionary do not include the ```"name.png"``` key, the translation will be filled with the empty line ```""```.

After inference, the metric calculation script shall compare the ```prediction_HTR.json``` and ```true_HTR.json``` files, and then display the final value of the metric for this task.

The ```true_HTR.json``` file shall have the following format:  ```{"0.txt": "<correct text in the picture>" , "1.txt": "<correct text in the picture>" , ... }```. Keys shall be represented by respective names of files from the images folder, while values shall be represented by the correct translation of a line in the respective image.

# Subtask 4 - Visual Question Answering

## Description

The objective is to give a text answer to a question about the image. The model input shall be the image and a text question related to it, while the output should be the answer to the question in the text format. For example, in this case, an answer to the question “What are the moustache made of?” may be represented by the word “bananas”:

![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/bananas.png)

The special feature of this task is that questions are not homogenous: a suitable answer may consist of several words or be one-syllable (yes/no answer) or represent a number. It is understood that it is necessary to give only one answer for one question.

Questions may be whether in English or in Russian. It is supposed that the answer language corresponds to the question language, unless the question refers to the text in the picture (for example, “What is written on a t-shirt?”) – it this case, the answer should be in the same language as the text on the image.

## Data

**Train.** It is suggested that a training set should be represented by a part of the train dataset [VQA v2](https://visualqa.org/download.html): it shall include questions in English (*Training questions 2017 v2.0* file), images from the COCO dataset, for which these questions are asked (*Training images* file), as well as annotations - answers to questions (*Training annotations 2017 v2.0* file).

**Test public.** The public test dataset consists of questions in both Russian and English: the Russian part represents translated 10,000 first samples from the validation part of the VQA v2 dataset, while the English one represents other 10,000 examples from the same dataset taken in the original form.

**Test private.** The private test dataset is hidden from participants. Its format is similar to the public test set and contains questions in Russian and English.

## Quality metric

The quality will be evaluated using the **accuracy** metric. It reflects the percentage of correct matches for the pairs of predicted and correct answers, i.e. the percentage of matching answers (the model predicts an answer matching the true one) to the total number of answers. This metric varies from 0 to 1, where 0 is the worst value and 1, the best one:

![image](https://latex.codecogs.com/svg.image?\textrm{accuracy}&space;=\frac{&space;\textrm{True&space;answers}}{&space;\textrm{All&space;answers}})

## Solution format

Participants should create an archive with a trained model and a set of scripts for model prediction. The participant shall upload this archive to the competition platform. Then, the archive shall be unzipped to a docker container, while the system shall add the data for prediction to the container space. Such data shall include:

* The ```images``` folder.  It is a set of images, to which the answers refer. It contains files in the following format: ```0.jpg, 1.jpg ...```.
* The ```questions.json``` file. It is a dictionary in the following format: ```{ "0": {"image_id": "1.jpg", "question": "Where is he looking?"} , ... }```. Keys shall be represented by sample indices, while values shall be represented by a dictionary with "image_id" (value: name of a file from the images folder) and "question" (value: text of a question for the respective image) fields. Questions may be asked in English or in Russian.


The participant’s model should make predictions for all questions and generate the ```prediction_VQA.json``` file. It is a dictionary in the following format: ```{ "0": "down" , ... }```. Keys shall be represented by sample indices, while values shall be represented by answers to the respective questions predicted by the model.

After inference, the metric calculation script shall compare the ```prediction_VQA.json``` and ```true_VQA.json``` files, and then display the final value of the accuracy metric.

# Integral metric

The final score shall be composed of scores for subtasks:

![image](https://latex.codecogs.com/svg.image?\textrm{S}&space;=&space;\textrm{S}_{1}&space;&plus;&space;\textrm{S}_{2}&space;&plus;&space;\textrm{S}_{3}&space;&plus;&space;\textrm{S}_{4},)

where S is a final score of the participant, S<sub>1</sub> is a score for the Code2code translation subtask, S<sub>2</sub> is a score for the Zero-shot object detection subtask, S<sub>3</sub> is a score for the Handwritten Text Recognition subtask, S<sub>4</sub> is a score for the Visual Question Answering subtask.

Scores for each subtask will take values from 0 to 1 (the only exception is the CodeBLEU metric used to evaluate Code2code translation: it may take values within the range from 0 to 100 – with a view to normalize it, the metric will be multiplied by 0.01) – therefore, the lowest value of the final score will be 0, while the highest one will be 4. Calculation of scores for each subtask shall be rounded to the third decimal place. The final score value shall serve as the basis for generating a leaderboard for the Fusion Brain Challenge task.

# Prize pool

For each winning place there is a fixed prize (FIX). The bonus shall depend on the winners’ final scores, but not exceeding the difference between the maximum (MAX) and fixed value. In each task, it is necessary to outperform the baseline. In each task, it is necessary to surpass baseline metrics for each subtask, while the integral metric should surpass the integral baseline metric at least by 0.15 (δ).

The prize amount shall be calculated according to the following formula:

![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/prize.png)

, where S is the participant’s final score, S<sub>baseline</sub> is the baseline final score, δ = 0.15 is the lowest value for surpassing the baseline final score, while the α coefficient shall depend on the place in leaderboard (Top-3 solutions) and be calculated as follows:

![image](https://latex.codecogs.com/svg.image?\alpha_{place}&space;=&space;\frac{\textrm{MAX}_{place}&space;-&space;\textrm{FIX}_{place}}{2.3&space;-&space;(\textrm{S}_{baseline}&space;&plus;&space;\delta)},)

where α<sub>place</sub> is a coefficient for the first, second and third places in the leaderboard (the place index means a place in the final leaderboard).  MAX<sub>place</sub> is the highest prize for Top-3 solutions in the leaderboard with S ≥ 2.3 (MAX<sub>1</sub> = RUB 3 million, MAX<sub>2</sub> = RUB 1.5 million, MAX<sub>3</sub> = RUB 0.8 million). FIX<sub>place</sub> is a fixed prize for top solutions in the leaderboard with (S<sub>baseline</sub> + δ) ≤ S < 2.3 (FIX<sub>1</sub> = 1, FIX<sub>2</sub> = 0.5, FIX<sub>3</sub> = 0.2). The α<sub>place</sub> coefficient shall be calculated only for cases, where S<sub>baseline</sub> + δ ≤ S < 2.3 (see the table above).
    
![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/prize_plot_en.png)
    
**First place**: from RUB 1,000,000 to RUB 3,000,000 (depending on the quality of participant’s solution)  
**Second place**: from RUB 500,000 to RUB 1,500,000 (depending on the quality of participant’s solution)  
**Third place**: from RUB 200 000 to RUB 800,000 (depending on the quality of participant’s solution)  
    
**Additional categories**:
    
RUB 300,000 for the first place in the Code2code translation subtask  
RUB 300,000 for the first place in the Zero-shot object detection subtask  
RUB 300,000 for the first place in the Handwritten Text Recognition subtask  
RUB 300,000 for the first place in the Visual Question Answering subtask
    
[Link to the Rules of the "Artificial Intelligence Journey Contest"](https://api.dsworks.ru/dsworks-transfer/api/v1/public/file/rules.pdf/download)  
[Terms of use](https://api.dsworks.ru/dsworks-transfer/api/v1/public/file/terms_of_use.pdf/download)
