# Fusion Brain Challenge

The current task suggests developing a single multitask model, which would successfully complete such multimodality subtasks as **Code2code translation (С2С), Zero-shot object detection (OD), Handwritten Text Recognition (HTR), Visual Question Answering (VQA)** and be able to break an integral metric of baseline proposed by the Arranger, as well as metrics for each subtask.

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

An argument accepted by the script for model inference should be represented by a path to the folder with the content to be used for prediction. Let us assume that the argument shall be represented by the ```fusion_brain``` folder. Names of ```fusion_brain``` subfolders shall correspond to names of subtasks to be completed by the single model. Each subfolder (HTR, OD, VQA, C2C) shall have the content needed for predictions. Below we discuss the structure of this content for each subtask in more detail.

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

Cases where a source language and a target language have different type systems are particularly problematic. It is that very class that our solution belongs to: we should translate from a statically-typed language (Java) to a dynamically-typed one (Python). The input is a Java function and the output should be the similar function in Python.

## Data

**Train.** It is suggested that the training dataset should be represented by train (5,937 programmer tasks) and val (845) parts of the [AVATAR](https://github.com/wasiahmad/AVATAR) parallel corpus consisting of pairs of similar functions/programs, one of which is written in Java, while the other is written in Python. Due to the fact that the dataset contains 1-25 candidate solutions in both programming languages, it shall be possible to generate 1-25 parallel examples for each task. The dataset authors suggest choosing no more than three solutions for each language - therefore, there will be no more than nine training examples for one problem. We suggest using a [dataset](https://dsworks.s3pd01.sbercloud.ru/aij2021/%D0%A12%D0%A1_translation/CodeToCodeTranslation_train_java_python.jsonl) generated in this very manner.

The file have the jsonl format with java and python fields:
```
{"java":"import java . util . Scanner ; \u00a0 public class A1437 { \u00a0 public static void main ( String [ ] args ) { Scanner in = new Scanner ( System . in ) ; int T = in . nextInt ( ) ; for ( int t = 0 ; t < T ; t ++ ) { int L = in . nextInt ( ) ; int R = in . nextInt ( ) ; boolean possible = R < 2 * L ; System . out . println ( possible ? \" YES \" : \" NO \" ) ; } } \u00a0 }\n","python":"t = int ( input ( ) ) NEW_LINE ans = [ ] NEW_LINE for i in range ( t ) : l , r = [ int ( x ) for x in input ( ) . split ( ) ] NEW_LINE if ( 2 * l ) > r : NEW_LINE INDENT ans . append ( \" YES \" ) else : NEW_LINE ans . append ( \" NO \" ) NEW_LINE DEDENT for j in ans : print ( j ) NEW_LINE\n"}
```
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

where BP is a penalty for too short translations, which shall be calculated as the number of tokens in the model-suggested translation divided by the number of tokens in the reference sequence; and the second part of the formula is a geometric mean of n-grams adjusted accuracy:

![image](https://latex.codecogs.com/svg.image?\textrm{p}_{n}&space;=&space;\frac{\sum_{C&space;\in&space;Candidates}&space;\sum_{n\text{-}gram&space;\in&space;C}&space;Count_{clip}&space;\textrm{(n-gram)}}{\sum_{C\textquotesingle&space;\in&space;Candidates}&space;\sum_{n\text{-}gram\textquotesingle&space;\in&space;C\textquotesingle}&space;Count&space;\textrm{(n-gram\textquotesingle)}})

for n-grams with the length from 1 to N, multiplied by corresponding positive weights w<sub>n</sub> that in total make 1.

* As opposed to the standard BLEU metric, the BLEU<sub>weight</sub> metric calculates the accuracy of n-grams coincidence by using the weight factor (![image](https://latex.codecogs.com/svg.image?\mu_{n}^{i})) with its value higher for keywords in a programming language than for any other tokens:

![image](https://latex.codecogs.com/svg.image?\textrm{p}_{n}&space;=&space;\frac{\sum_{C&space;\in&space;Candidates}&space;\sum_{i=1}^{l}&space;\mu_{n}^{i}&space;Count_{clip}&space;(C(i,&space;i&plus;n))}{\sum_{C\textquotesingle&space;\in&space;Candidates}&space;\sum_{i=1}^{l}&space;\mu_{n}^{i}&space;Count&space;(C\textquotesingle(i,&space;i&plus;n))},)

where C(i, i+n) is an n-gram starting in the i place and ending in the i+n place, Count<sub>clip</sub> has the same meaning as that in the standard BLEU metric, it is the maximum number of n-grams found in both translated code and reference set. The keywords list is pre-determined for each programming language.

* For a source code, its syntax structure could be expressed as an abstract syntax tree (AST), thus enabling to compare the translated and reference functions on the level of subtrees generated by an AST parser. Since we are interested in the syntax, the AST leaves containing variables could be omitted. Match<sub>ast</sub> shall be calculated according to the following formula:

![image](https://latex.codecogs.com/svg.image?\textrm{Match}_{ast}&space;=&space;\frac{\textrm{Count}_{clip}(\textrm{T}_{cand})}{\textrm{Count}(\textrm{T}_{ref})},)

where Count(T<sub>ref</sub>) is a total number of subtrees in the reference code, Count<sub>clip</sub>(T<sub>cand</sub>) is a number of subtrees in the translated code that have matched the subtrees of reference functions. This metric allows assessing the quality of a code translated in terms of its syntax.

* The translated code and reference code are also compared by their semantics, using data flows (```Guo et al., 2020```) when the source code is represented as a graph with its points being variables and its sides, the ‘genetic’ relations between the points (denoting the origin of each variable value). The Match<sub>df</sub> metric shall be calculated according to the following formula:

![image](https://latex.codecogs.com/svg.image?\textrm{Match}_{df}&space;=&space;\frac{\textrm{Count}_{clip}(\textrm{DF}_{cand})}{\textrm{Count}(\textrm{DF}_{ref})},)

where Count(DF<sub>ref</sub>) is a total number of data flows in the reference code; Count<sub>clip</sub>(DF<sub>cand</sub>) is a number of data flows in the translated code that have matched the reference code.

# Solution format

Participants should create an archive with a trained model and a set of scripts for model prediction. The participant shall upload this archive to the competition platform. Then, the archive shall be unzipped to a docker container, while the system shall add the data for prediction to the container space. Such data shall include:

* The ```requests.json``` file. It is a dictionary in the following format: ```{ "0": "import java . util . Scanner ; ..." , ... }```. Keys shall be represented by example indices, while values shall be represented by lines of functions/programs in Java that should be translated into Python.

The participant’s model should translate all examples from the requests.json file and generate the ```prediction_С2С.json``` file. It is a dictionary in the following format: ```{ "0": "def find ( x , par ) : NEW_LINE INDENT if par [ x ] == x : ..." , ... }```. Keys shall be represented by example indices, while values shall be represented by translations of functions/programs into Python. Please, pay attention to the fact that since Python uses indentations to identify logic blocks in codes, the line of translation into Python includes such special tokens as ```INDENT```, ```DEDENT```. 

After inference, the metric calculation script shall compare the ```prediction_С2С.json``` and ```true_С2С.json``` files, and then display the final value of the CodeBLEU metric.
