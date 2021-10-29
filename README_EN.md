# Fusion Brain Challenge
## Update

**27.10**:

* In the **Sample submission** section, we added a reference to the current sample submission and references to examples of input/output data.
* We specified the subtask **HTR** by reference from section **Sample submission**. You can download a test dataset containing the collected and labeled images with text from school notebooks (in Russian and English).  Note that test public, in addition to standard ones, contains examples with a vertical orientation (rotated by 90 degrees). The data distribution in the handwritten dataset of notebooks in English differs from the data distribution of the **IAM** dataset, so we advise applying different types of normalization, augmentation - and learn not only by **IAM**.
* In section **Data** VQA task added a reference to the dataset, a subsample of VQA v2 (train) that intersects with the Visual Genome training sample - these are questions with answers (in Russian and English) for 33,821 images.

**28.10**:

* We remind that there are images rotated 90 degrees counterclockwise in the public test. In a private test, however, there won't be such images. At this [link](https://dsworks.s3pd01.sbercloud.ru/aij2021/sample_data/rotate.zip) you can find an archive that contains the file `rotate.json` - a list of images from the public test that needs to be rotated 90 degrees clockwise. An example of using this file is shown in the `rotate.py` script inside the archive. You can use this json in your solution script to rotate the desired images. This will only affect the value of the public leaderboard metric; it is useless for a private test, as there won't be vertical images. In the dataset of Russian notebooks presented for training, as in the IAM, there are no rotated images, so it makes no sense to specifically teach the model to rotate images.


## General task description
The current task suggests developing a single multitask model that would successfully solve such multimodality subtasks as **Code2code Translation (С2С), Handwritten Text Recognition (HTR), Zero-shot Object Detection (zsOD), Visual Question Answering (VQA)** and be able to surpass the minimum value of the integral metric established by the Arranger, as well as the minimum values of the metrics for each of the sub-tasks. 

We provide [a concept of a single model](https://colab.research.google.com/drive/1YAkxWG0dRKPtqy9CZxFPvCNCCXvMGr65?usp=sharing%22%20target%3D%22_parent%22%3E%3Cimg%20src%3D%22https%3A%2F%2Fcolab.research.google.com%2Fassets%2Fcolab-badge.svg%22%20alt%3D%22Open%20In%20Colab) that is trained on several tasks related to different modalities (visual, audio and text). The concept is inspired by an article ["Pretrained Transformers as Universal Computations Engines"](https://arxiv.org/pdf/2103.05247.pdf) (```Lu et al., 2021```) that examines the ability of pretrained language models based on the Transformer architecture to form qualitative representations of arbitrary data sequences – thus, generalizing to other modalities with minimal finetuning. The basis of the architecture proposed in the concept is the pretrained GPT-2 language model; experiments are carried out both with a "frozen" model (Frozen Pretrained Transformer), and with a model in which all layers are trained on three modalities simultaneously.

**It is recommended to get acquainted with the [review on multitask and multimodal architectures](https://github.com/sberbank-ai/fusion_brain_aij2021/blob/main/Papers%20on%20multitask%20%26%20multimodal%20models_en.md).**

In order for the model presented by the team/participant to be considered as multitask, it is necessary and sufficient to meet the following criteria:

1) shared weights should be **at least 30%** of all model parameters: *if ![image](https://latex.codecogs.com/svg.image?\color{Blue}N) is the total number of parameters of the models that solve 4 subtasks, and ![image](https://latex.codecogs.com/svg.image?\color{Blue}M) is the number of common parameters of these models (that is, they are identical both in value and architecturally), then it is necessary that ![image](https://latex.codecogs.com/svg.image?\color{Blue}M/N\geqslant&space;0.3)*

2) common parameters should not be purely nominal - on the contrary, they should be used in a **meaningful** way during the prediction of the model and have a **beneficial** effect on model’s quality.

If at least one of the criteria above is not met, the model is considered to solve the subtask (or subtasks) separately.

Uploading solutions to the Competition platform will become available from **07/10/2021**.

## General solution format

### Container content

Participants should create a zip archive with a trained model and a set of scripts for model prediction. The solutions are run in an isolated environment using Docker. Time and resources during testing are limited. The participant does not need to dive into Docker technology.

The archive root must contain the `metadata.json` file containing the following:
```
{
    "image": "cr.msk.sbercloud.ru/aijcontest2021/fusion:0.0.1",
    "entrypoint": "python /home/jovyan/run.py"
}
```

Where ```image``` is a field with the docker image name, in which the solution will be run, ```entrypoint``` is a command that runs the inference script. For solution, `/home/jovyan` will be the current directory. 

To run solutions, you can use the following environment:

* ```cr.msk.sbercloud.ru/aijcontest2021/fusion:0.0.1``` - [Dockerfile и requirements](https://drive.google.com/file/d/1JGv-khEGv8tFD6wAiLJIP6qocd0CUZIJ/view?usp=sharing) for this image.

If necessary, you can create your custom image by adding the necessary software and libraries to it ([instructions for creating docker images](https://github.com/sberbank-ai/no_fire_with_ai_aij2021/blob/main/sbercloud_instruction.md)); to use it you need to publish it on ```sbercloud```. Custom images must be inherited from base `sbercloud` images ([base images](https://docs.sbercloud.ru/aicloud/mlspace/concepts/environments__basic-images-for-training.html)). When creating a custom image, you must assign it an individual name and tag (for example ```my_custom_fusionchallenge: 0.0.5```).

### Data Structure

The `input` folder is placed in the container. Names of ```input``` subfolders shall correspond to names of subtasks to be completed by the single model. Each subfolder (C2C, HTR, zsOD, VQA) shall have the content needed for predictions.

The data structure is as follows:

* input
  * C2C 
    * requests.json
  * HTR
    * images 
  * zsOD
    * images
    * requests.json
  * VQA
    * images
    * questions.json

The single model should generate predictions for each subtask in the ```prediction_{TASK_NAME}.json``` format, i.e. there should be four files after the model inference: ```prediction_C2C.json, prediction_HTR.json, prediction_zsOD.json, prediction_VQA.json```. These files should be located in the `output` folder (absolute path: `/home/jovyan/output`).

The structure of the model prediction directory should be as follows:

* output
  * prediction_C2C.json
  * prediction_HTR.json
  * prediction_zsOD.json
  * prediction_VQA.json

After that, correct answers in the ```true_{TASK_NAME}.json``` format shall be added to the container and a script for calculation of metrics for each subtask shall be run. The final metric shall be calculated as a sum of metrics for each subtask (see below).

### Baseline

In  `fb_baseline` folder is a basic solution for all four subtasks. This solution is based on the multimodal model concept from `fb_concept`. In `fb_baseline/FBC_baseline.ipynb` notebook is a code with dataset formation, model architecture, and learning logic. In `fb_baseline/fb_utils` folder is auxiliary set of scripts.

* `BLEU.py` и `c2c_eval.py` - scripts that contain auxiliary functions for inference and metric computation in the C2C task
* `detection_vqa.py` - script that contains loss functions, feedforward and cross-attention layers, and auxiliary functions for zsOD and VQA tasks.
* `download.py` - script downloading files for training.
* `handwritten.py` - script with auxiliary functions for the HTR task.
* `metrics.py` - calculating the metric of all four subtasks.

The following figure provides a diagram of the proposed architecture:

<p align="center">
  <img src="fb_baseline/FusionBrain.png" width="70%">
</p>

This [link](https://www.youtube.com/watch?v=cNEsBWXmJqY&t=5982s) you can see the webinar with a detailed analysis of basilene.

### Sample submission

This [link](https://dsworks.s3pd01.sbercloud.ru/aij2021/baseline/sberai_baseline_ooc.zip) is for the `sberai_baseline_ooc.zip` archive, which contains sample submission. You can find quality metrics of this solution in the leaderboard of the competition by the team's name `sberaiooc`. The following files in the archive are required to generate model predictions:

 * `metadata.json` - required file for every solution; it must contain the paths to the image and the model inference script
 * `run.py` - main model inference script
 * `last.pt` - model weights that are loaded during the execution of the `run.py` script
 * `utils` - folder with auxiliary scripts for `run.py`. In the case of a baseline, it contains two files:
     * `dataset.py` - code for `DatasetRetriever` class and `fb_collate_fn` function
     * `fb_model.py` - code for creating the model class
 * `gpt_init` - folder with necessary files for `GPT2Tokenizer` and `GPT2Model` initialization
 * `fb_utils` - additional set of scripts; it is analogous (with some exceptions) to the same subfolder from `fb_baseline` folder in this repository.

Data examples (we advise you to pay attention to `input/HTR/images`: in addition to Russian, there are examples in English)

* [input](https://dsworks.s3pd01.sbercloud.ru/aij2021/sample_data/input.zip)  - samples of input data for each of the tasks;
* [output](https://dsworks.s3pd01.sbercloud.ru/aij2021/sample_data/output.zip) - samples of model predictions for files from a folder `input`. These are random predictions and only show the correct format that is expected from the participant's model;
* [true](https://dsworks.s3pd01.sbercloud.ru/aij2021/sample_data/true.zip) - samples of files with correct answers for each problem, the predictions from `output` are compared with them during the calculation of the metrics.

## Limitations

A Participant or a Team of Participants can submit a maximum of 3 (three) submission per day. Only valid attempts that have received a numerical estimate are taken into account.

The solution container will be run under the following conditions:

* 100 GB RAM
* 3 vCPU
* 1 GPU Tesla V100 (32 GB)
* Time for performance: 90m
* Offline solution
* Maximum size of your solution archive compressed and decompressed: 10 GB
* Maximum size of the Docker image used: 15 GB.

We provide participants with the ability to access the computational resources of Christophari to train the model. The number of resources is limited. To gain access, you should send an application to `Christofari_AIJContest_2021@sberbank.ru` with a description of how exactly it is planned to use computational resources.

## Quality metrics

The corresponding quality metric for each of the subtasks could be found [here](https://github.com/sberbank-ai/fusion_brain_aij2021/blob/main/Metrics_en.md).

# Subtask 1 - Code2code Translation

## Description

A task of translation from one programming language into another is a standard part of the wide-ranging repertoire of ML4Code. As of today, there are several alternate solutions – in line with both supervised learning, where a training dataset is represented by a parallel corpus (baseline model of the [CodeXGLUE](https://arxiv.org/pdf/2102.04664.pdf) benchmark with CodeBERT as encoder in the encoder-decoder architecture, ```Lu et al., 2021```), and unsupervised one, including pretraining of cross-lingual language model on monolingual corpora ([TransCoder](https://arxiv.org/pdf/2006.03511.pdf), ```Lachaux et al., 2020```). 

Cases where a source language and a target language have different type systems are particularly problematic. It is that very class that our subtask belongs to: we should translate from a statically-typed language (Java) to a dynamically-typed one (Python). The input is a Java function and the output should be the similar function in Python.

## Data

**Train.** It is suggested that the training dataset should be represented by train (5,937 programmer tasks) and val (845) parts of the [AVATAR](https://github.com/wasiahmad/AVATAR) parallel corpus consisting of pairs of similar functions/programs, one of which is written in Java, while the other is written in Python. Due to the fact that the dataset contains 1-25 candidate solutions in both programming languages, it shall be possible to generate 1-25 parallel examples for each task. The dataset authors suggest choosing no more than three solutions for each language - therefore, there will be no more than nine training examples for one problem. We suggest using a [dataset](https://dsworks.s3pd01.sbercloud.ru/aij2021/%D0%A12%D0%A1_translation/CodeToCodeTranslation_train_java_python.jsonl) generated in this very manner.

The file have the jsonl format with "java" and "python" fields:
```
{"java":"import java . util . Scanner ; \u00a0 public class A1437 { \u00a0 public static void main ( String [ ] args ) { Scanner in = new Scanner ( System . in ) ; int T = in . nextInt ( ) ; for ( int t = 0 ; t < T ; t ++ ) { int L = in . nextInt ( ) ; int R = in . nextInt ( ) ; boolean possible = R < 2 * L ; System . out . println ( possible ? \" YES \" : \" NO \" ) ; } } \u00a0 }\n","python":"t = int ( input ( ) ) NEW_LINE ans = [ ] NEW_LINE for i in range ( t ) : l , r = [ int ( x ) for x in input ( ) . split ( ) ] NEW_LINE if ( 2 * l ) > r : NEW_LINE INDENT ans . append ( \" YES \" ) else : NEW_LINE ans . append ( \" NO \" ) NEW_LINE DEDENT for j in ans : print ( j ) NEW_LINE\n"}
```

To create a parallel training corpus, you can also use [CodeNet](https://github.com/IBM/Project_CodeNet), which contains solutions in 4 languages (C++, C, Python and Java) to 4,000 programming problems, extracted from two online judge web sites: AtCoder (the AVATAR dataset also includes solutions from this resource for some tasks) and AIZU Online Judge. For the convenience of participants, we provide an [archive](https://dsworks.s3pd01.sbercloud.ru/aij2021/%D0%A12%D0%A1_translation/CodeNet_accepted_java_python.tar.gz) (the full data is in the repository https://developer.ibm.com/technologies/artificial-intelligence/data/project-codenet/) containing solutions from CodeNet written in Java and Python, broken down by problems. However, it should be taken into account that the solutions of one programming problem in different languages are, at least, type IV clones (preserving source code semantics, but having considerable differences in syntax), but they are not guaranteed to be identical to each other, adjusted for peculiarities of languages (literal translation).

**Test public.** The public leaderboard shall be generated according to the results of model prediction check based on a test set (1,699 samples) from the AVATAR dataset.

**Test private.** The private test dataset is hidden from participants. Its format is similar to the public test set.

## Solution format

Data for prediction related to this subtask shall include:

* The ```requests.json``` file. It is a dictionary in the following format: ```{ "0": "import java . util . Scanner ; ..." , ... }```. Keys shall be represented by sample indices, while values shall be represented by lines of functions/programs in Java that should be translated into Python.

The participant’s model should translate all examples from the requests.json file and generate the ```prediction_С2С.json``` file. It is a dictionary in the following format: ```{ "0": "def find ( x , par ) : NEW_LINE INDENT if par [ x ] == x : ..." , ... }```. Keys shall be represented by sample indices, while values shall be represented by translations of functions/programs into Python. Please, pay attention to the fact that since Python uses indentations to identify logic blocks in codes, the line of translation into Python includes such special tokens as ```INDENT```, ```DEDENT```. 

After inference, the metric calculation script shall compare the ```prediction_С2С.json``` and ```true_С2С.json``` files, and then display the final value of the CodeBLEU metric.

# Subtask 2 - Handwritten Text Recognition

## Description

Participants are given the task to recognize a handwritten text in the picture. The model input is an image with a text written in English or Russian. The model output should be a text line corresponding to the image content (in this case - the line “последовал”):

![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/htr_posledoval.png)

## Data

**Train.** We provide a training [dataset] (https://dsworks.s3pd01.sbercloud.ru/aij2021/htr/train.zip) consisting of a manually gathered and processed collection of school copybooks. Images in this dataset are represented by individual words in a text written with the Cyrillic characters on a copybook page. As for the handwritten words in English, we recommend to use a popular dataset called [IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).

**Test public.** The public leaderboard shall be calculated with regard to the copybook dataset containing texts in Russian and English (14,973 samples).

**Test private.** The private test dataset is hidden from participants. It is also a dataset for text recognition in a format similar to training dataset.

## Solution format

Data for prediction related to this subtask shall include:

* The ```images``` folder.  It is a set of images to make predictions for. It contains files in the following format: ```0.png, 1.png ...```. Each file contains graphic images of characters to be translated into text characters (text lines).

The participant’s model should make predictions for all images from the images folder and generate a ```prediction_HTR.json``` file. It is a dictionary in the following format:  ```{"0.png": "<predicted text in the picture>" , "1.png": "<predicted text in the picture>" , ... }```. Keys shall be represented by respective names of files from the images folder, while values shall be represented by predicted lines in the respective images. If there is no prediction for a name.png file with the image, i.e. keys of the ```prediction_HTR.json``` dictionary do not include the ```"name.png"``` key, the translation will be filled with the empty line ```""```.

After inference, the metric calculation script shall compare the ```prediction_HTR.json``` and ```true_HTR.json``` files, and then display the final value of the metric for this task.

The ```true_HTR.json``` file shall have the following format:  ```{"0.png": "<correct text in the picture>" , "1.png": "<correct text in the picture>" , ... }```. Keys shall be represented by respective names of files from the images folder, while values shall be represented by the correct translation of a line in the respective image.


# Subtask 3 - Zero-shot Object Detection

## Description

* It is necessary to determine the correct description of the object depicted in the photo (or objects, if there are several). For example, there can be such entities/objects on the photo that could be described in natural language as "green apple lying on the ground”, "man jumping over fire hydrant”, "woman in shorts".

* At the same time, it is necessary to determine the location and scale of each object shown in a photo. The object location shall be described with the so-called bounding box (bbox). This is a rectangle to be drawn most accurately around the object. The rectangle position shall be set with four numbers – X, Y, W, H, where:

    * X is a horizontal coordinate of the top left corner
    * Y is a vertical coordinate of the top left corner
    * W is the rectangle width
    * H is the rectangle height

For each object in a photo, the model predictions should be represented by bbox coordinates and a class (a description in natural language) for each object in the photo.  An example of the result of object detection model work is shown in the next picture:

![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/vg_region_description.png)

Within the framework of our competition, the task is defines as zero-shot object detection. Zero-shot in the task description means that the model should be able to succesfully make predictions on a dataset completely differing from the training one. A standard object detection model is expected to predict one class out of a limited set of classes determined during the model training. A zero-shot model is expected to detect classes not found in the training set.

The set of possible classes for each image is fed into a model as a query. The query may contain classes in Russian or in English.

At the prediction stage, the model input shall contain two entities: an image and a natural-language query. The query is formatted as a list of text lines (descriptions) – classes to search from. Example: ```"red apple hanging on a branch", "bald man", "girl feeding an elephant"```. The query contains both correct descriptions (related to objects actually present in the picture) and some incorrect ones. Their combination makes a single search space for the model. The model should output a list of predicted classes with the corresponding bounding box coordinates.

## Data

**Train.** It is suggested that training should be based on a popular dataset called [MS-COCO](https://cocodataset.org/#download), which contains images (*2017 Train images* file) and their corresponding annotations (*2017 Train/Val annotations* file).

It is also worth using the [VisualGenome dataset](https://visualgenome.org/api/v0/api_home.html), except for the [images](https://dsworks.s3pd01.sbercloud.ru/aij2021/zsOD/images_ids_Visual_Genome_to_exclude.json) included in the public test dataset (so that the results of the public leaderboard are indicative for the participants).

We also provide a dataset that is a VisualGenome (train) subsample that overlaps with the VQA v2 training set - these are region descriptions (on average 10 descriptions per image) with corresponding bounding boxes for 33,821 images; half of the samples are [in English](https://dsworks.s3pd01.sbercloud.ru/aij2021/zsOD/vg_intersection_eng.json), the other half [in Russian](https://dsworks.s3pd01.sbercloud.ru/aij2021/zsOD/vg_intersection_rus.json) (obtained through machine translation). This [link](https://dsworks.s3pd01.sbercloud.ru/aij2021/zsOD/vgid_to_cocoid.json) contains a mapping from image IDs in VisualGenome to COCO IDs (and VQA v2, respectively).

**Test public.** The public test dataset is generated from a part of the VisualGenome dataset (1,000 samples); a set of classes in it is hidden from participants. Region descriptions from VisualGenome are used as positive classes (descriptions are subjected to normalization: conversion to lower case, removal of non-printable characters, etc.; boxes related to one entity are combined under a single description); negative classes are formed by replacing some objects/attributes in the description with those that are not in the photo: for example, the `chair is gray` is replaced by the `chair is pink`, `cushion on the end of the sofa` is replaced by `cushion on the end of the table`. Also, descriptions of objects belonging to the same domain as the correct classes are used as negative examples: if the photo shows a street, then as negative examples there may be, for example, such descriptions as `tall green bricks wall`, `shingled home in distance`, `food stand in the street` (provided, of course, that the described objects are not in the photo).

**Test private.** The private test dataset is hidden from participants, just as a set of classes in it.

## Solution format

Data for prediction related to this subtask shall include:

* The ```images``` folder.  It is a set of images to make predictions for. It contains files in the following format: ```0.jpg, 1.jpg ...```.
* The ```requests.json``` file. It is a dictionary in the following format: ```{ "0.jpg": ["red apple hanging on a branch", "bald man", "girl feeding an elephant"] , ... }```. Keys shall be represented by respective names of files from the images folder, while values shall be represented by a list of classes to be detected in the respective image (query). As we said before, the list of classes (descriptions in natural language) may be whether in Russian or in English. Therefore, the query contains the list of classes to search from. **The query contains both correct descriptions belonging to objects actually present in the image and some incorrect descriptions (there no respective objects in the picture)**.

The participant’s model should make predictions for all images from the images folder and generate a ```prediction_OD.json``` file. It is a dictionary in the following format: ```{"0.jpg": {"red apple hanging on a branch": [[473.07, 395.93, 38.65, 28.67]], "bald man": [], "girl feeding an elephant": [[0.0, 101.15, 452.3, 319.43], [10.0, 123.0, 15.0, 22.0]]}, ...}```. Keys shall be represented by names of files from the `images` folder, while values shall be represented by dictionaries, the keys in which are the names of classes suggested in `requests.json` for searching on the corresponding image, and the values, in turn, are model predictions for the corresponding class on this image. The predictions for each class inside each image should contain the coordinates of the bounding boxes: the key "name of file with the image" shall be used for showing a nested dictionary of classes. Inside each such dictionary, by the class name key there is a nested list of the format ```[[xmin, ymin, w, h]]```. The format of one element in the list: ```[473.07, 395.93, 38.65, 28.67]``` (four items separated by commas) – bbox coordinates in the `xywh` format. The nested list may contain the unlimited number of elements - these are all bboxes that the model predicted for a given class in the particular image.

The dictionary of correct answers `true_OD.json`, which will be used to evaluate the quality of the model in the docker container, has the following format: `{img_name: {class_name1: [[xmin, ymin, w, h]], class_name2: [], class_name3: [[xmin, ymin, w, h], [xmin, ymin, w, h]]}, ...}`. If an empty list is found by the class key, this means that this class in the request from `requests.json` is negative, that is, the described object is absent on the image. The model should not predict anything for the given class as well, that is, an empty list `[]` should be passed.

Then, the system shall compare the file with predictions with the ```true_OD.json``` file containing correct answers, and display the final F1-score metric.

# Subtask 4 - Visual Question Answering

## Description

The objective is to give a text answer to a question about the image. The model input shall be the image and a text question related to it, while the output should be the answer to the question in the text format. For example, in this case, an answer to the question “What are the moustache made of?” may be represented by the word “bananas”:

![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/bananas.png)

The special feature of this task is that questions are not homogenous: a suitable answer may consist of several words or be one-syllable (yes/no answer) or represent a number. It is understood that it is necessary to give only one answer for one question.

Questions may be whether in English or in Russian. It is supposed that the answer language corresponds to the question language, unless the question refers to the text in the picture (for example, “What is written on a t-shirt?”) – it this case, the answer should be in the same language as the text on the image.

## Data

**Train.** It is suggested that a training set should be represented by a part of the train dataset [VQA v2](https://visualqa.org/download.html): it shall include questions in English (*Training questions 2017 v2.0* file), images from the COCO dataset, for which these questions are asked (*Training images* file), as well as annotations - answers to questions (*Training annotations 2017 v2.0* file).

We also provide a dataset that is a VQA v2 (train) subsample that overlaps with the Visual Genome training set - questions and answers for 33,821 images; for half of the images, questions are given [in English](https://dsworks.s3pd01.sbercloud.ru/aij2021/VQA/VQA_eng.zip) (90,359 questions), for the other half - [in Russian](https://dsworks.s3pd01.sbercloud.ru/aij2021/VQA/VQA_rus.zip) (88,761 questions obtained through machine translation).

**Test public.** The public test dataset consists of questions in both Russian and English: the Russian part is represented by translated samples from the first 10,000 examples from the validation subset of the VQA v2 dataset, while the English part is represented by samples from the second 10,000 examples from the same dataset taken in the original form. The total size of the test public dataset is 5,446 samples.

**Test private.** The private test dataset is hidden from participants. Its format is similar to the public test set and contains questions in Russian and English.

## Solution format

Data for prediction related to this subtask shall include:

* The ```images``` folder.  It is a set of images, to which the answers refer. It contains files in the following format: ```0.jpg, 1.jpg ...```.
* The ```questions.json``` file. It is a dictionary in the following format: ```{ "0": {"file_name": "1.jpg", "question": "Where is he looking?"} , ... }```. Keys shall be represented by sample indices, while values shall be represented by a dictionary with "file_name" (value: name of a file from the images folder) and "question" (value: text of a question for the respective image) fields. Questions may be asked in English or in Russian.


The participant’s model should make predictions for all questions and generate the ```prediction_VQA.json``` file. It is a dictionary in the following format: ```{ "0": "down" , ... }```. Keys shall be represented by sample indices, while values shall be represented by answers to the respective questions predicted by the model.

After inference, the metric calculation script shall compare the ```prediction_VQA.json``` and ```true_VQA.json``` files, and then display the final value of the accuracy metric.

# Integral metric

The final score of multitask model shall be composed of scores for subtasks:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{S}&space;=&space;\textrm{S}_{1}&space;&plus;&space;\textrm{S}_{2}&space;&plus;&space;\textrm{S}_{3}&space;&plus;&space;\textrm{S}_{4},)

where S is a final score of the participant, S<sub>1</sub> is a score for the Code2code translation subtask, S<sub>2</sub> is a score for the Zero-shot object detection subtask, S<sub>3</sub> is a score for the Handwritten Text Recognition subtask, S<sub>4</sub> is a score for the Visual Question Answering subtask.

Scores for each subtask will take values from 0 to 1 (the only exception is the CodeBLEU metric used to evaluate Code2code Translation: it may take values within the range from 0 to 100 – with a view to normalize it, the metric will be multiplied by 0.01) – therefore, the lowest value of the final score will be 0, while the highest one will be 4. Calculation of scores for each subtask shall be rounded to the third decimal place. The final score value shall serve as the basis for generating a leaderboard for the Fusion Brain Challenge task.

# Prize pool

The possible Prize depends on whether the proposed architecture is a single mulitask model (with at least 30% of shared weights – parameters which are common to all modalities) or a unitask model (solving one subtask).

For each winning place there is a fixed prize (FIX). The bonus shall depend on the winners’ final scores, but not exceeding the difference between the maximum (MAX) and fixed value. It is necessary to surpass the minimum values of the metrics established for each subtask — and, consequently, the minimum value of the integral metric.

The minimum values for each of the subtasks are the following:

![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/S_mins_en.png)

The minimum value of the integral score S<sub>min</sub> is calculated as follows:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{S}_{min}&space;=&space;\textrm{S}_{min}^{1}&space;&plus;&space;\textrm{S}_{min}^{2}&space;&plus;&space;\textrm{S}_{min}^{3}&space;&plus;&space;\textrm{S}_{min}^{4}&space;=&space;0.2&space;&plus;&space;0.6&space;&plus;&space;0.15&space;&plus;&space;0.35&space;=&space;1.3)

The prize amount shall be calculated according to the following formula:

![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/prize.png)

, where S is the participant’s final score, S<sub>min</sub> is the minimum value of the final score, while the α coefficient shall depend on the place in leaderboard (Top-3 solutions) and be calculated as follows:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\alpha_{place}&space;=&space;\frac{\textrm{MAX}_{place}&space;-&space;\textrm{FIX}_{place}}{2.3&space;-&space;(\textrm{S}_{baseline}&space;&plus;&space;\delta)},)

where α<sub>place</sub> is a coefficient for calculating the bonus for the first, second and third places in the leaderboard (α_1 = 2, α_2 = 1, α_3 = 0.6) for cases, where S<sub>min</sub> ≤ S < 2.3. MAX<sub>place</sub> is the highest prize for Top-3 solutions in the leaderboard with S ≥ 2.3 (MAX<sub>1</sub> = RUB 3 million, MAX<sub>2</sub> = RUB 1.5 million, MAX<sub>3</sub> = RUB 0.8 million). FIX<sub>place</sub> is a fixed prize for top solutions in the leaderboard with S<sub>min</sub> ≤ S < 2.3 (FIX<sub>1</sub> = RUB 1 million, FIX<sub>2</sub> = RUB 0.5 million, FIX<sub>3</sub> = RUB 0.2 million).
    
![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/prize_plot_en.png)

**Nominations related to the creation of a multitask model**:
    
**First place**: from RUB 1,000,000 to RUB 3,000,000 (depending on the quality of participant’s solution)  
**Second place**: from RUB 500,000 to RUB 1,500,000 (depending on the quality of participant’s solution)  
**Third place**: from RUB 200 000 to RUB 800,000 (depending on the quality of participant’s solution)  
    
**Additional nominations**:

The Arranger will evaluate the best solutions for each of the subtasks, for which the participant/team of participants who showed the best result will also be able to get a Prize, regardless of whether the presented model is multitask (solving all subtasks with a single architecture) or unitask (solving one subtask). For each of the subtasks, the best solution will be chosen based on the leaderboard for these subtasks; the presented model should surpass the minimum value of the metric for the corresponding subtask (the minimum values are described above).
    
* RUB 300,000 for the first place in the Code2code translation subtask  
* RUB 300,000 for the first place in the Zero-shot object detection subtask  
* RUB 300,000 for the first place in the Handwritten Text Recognition subtask  
* RUB 300,000 for the first place in the Visual Question Answering subtask
    
[Link to the Rules of the "Artificial Intelligence Journey Contest"](https://api.dsworks.ru/dsworks-transfer/api/v1/public/file/rules.pdf/download)    
[Terms of use](https://api.dsworks.ru/dsworks-transfer/api/v1/public/file/terms_of_use.pdf/download)
