# Quality metrics

## Subtask 1 - Code2code Translation

**CodeBLEU** is a metric suggested in ```Ren et al., 2020``` to adjust the BLEU metric, which is normally used to assess the translation quality for natural languages, to the source code translation assessment; such adjustment supplementing the n-grams comparison by the comparison of corresponding abstract syntax trees in the reference code and translated code (thus we consider the code syntax), as well as juxtaposing the data flows (thus we consider the code semantics).

CodeBLEU is a weighted combination of four components:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{CodeBLEU}&space;=&space;\alpha&space;\cdot&space;\textrm{BLEU}&space;&plus;&space;\beta&space;\cdot&space;\textrm{BLEU}_{weight}&space;&plus;&space;\gamma&space;\cdot&space;\textrm{Match}_{ast}&space;&plus;&space;\delta&space;\cdot&space;\textrm{Match}_{df},)

where BLEU is the standard BLEU metric ```Papineni et al., 2002```, BLEU<sub>weight</sub> is the weighted comparison of n-grams (with the tokens differing by their significance and the match of certain tokens for the translated and ‘golden’ functions is assigned more weight), Match<sub>ast</sub> is the comparison metric for abstract syntax trees of the translated code and of the reference code, and Match<sub>df</sub> reflects the similarity of data flows produced by hypotheses and correct functions.

Let us discuss each component of the metric in more detail.

* BLEU is based on calculation of n-grams found in the translation and the reference sequence calculated as follows:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{BLEU}&space;=&space;\textrm{BP}&space;\cdot&space;\textrm{exp}&space;(\sum_{n=1}^{N}&space;w_{n}&space;\log&space;p_{n}),)

where BP is a penalty for too short translations, which shall be calculated as the number of tokens in the model-suggested translation divided by the number of tokens in the reference sequence; and the second part of the formula is a geometric mean of a modified n-gram precision:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{p}_{n}&space;=&space;\frac{\sum_{C&space;\in&space;Candidates}&space;\sum_{n\text{-}gram&space;\in&space;C}&space;Count_{clip}&space;\textrm{(n-gram)}}{\sum_{C\textquotesingle&space;\in&space;Candidates}&space;\sum_{n\text{-}gram\textquotesingle&space;\in&space;C\textquotesingle}&space;Count&space;\textrm{(n-gram\textquotesingle)}})

for n-grams with the length from 1 to N, multiplied by corresponding positive weights w<sub>n</sub> that in total make 1.

* As opposed to the standard BLEU metric, the BLEU<sub>weight</sub> metric calculates the accuracy of n-grams coincidence by using the weight factor (![image](https://latex.codecogs.com/svg.image?\color{Blue}\mu_{n}^{i})) with its value higher for keywords in a programming language than for any other tokens:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{p}_{n}&space;=&space;\frac{\sum_{C&space;\in&space;Candidates}&space;\sum_{i=1}^{l}&space;\mu_{n}^{i}&space;Count_{clip}&space;(C(i,&space;i&plus;n))}{\sum_{C\textquotesingle&space;\in&space;Candidates}&space;\sum_{i=1}^{l}&space;\mu_{n}^{i}&space;Count&space;(C\textquotesingle(i,&space;i&plus;n))},)

where C(i, i+n) is an n-gram starting in the i place and ending in the i+n place, Count<sub>clip</sub> has the same meaning as that in the standard BLEU metric, it is the maximum number of n-grams found in both translated code and reference set. The keywords list is pre-determined for each programming language.

* For a source code, its syntax structure could be expressed as an abstract syntax tree (AST), thus enabling to compare the translated and reference functions on the level of subtrees generated by an AST parser. Since we are interested in the syntax, the AST leaves containing variables could be omitted. Match<sub>ast</sub> shall be calculated according to the following formula:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{Match}_{ast}&space;=&space;\frac{\textrm{Count}_{clip}(\textrm{T}_{cand})}{\textrm{Count}(\textrm{T}_{ref})},)

where Count(T<sub>ref</sub>) is a total number of subtrees in the reference code, Count<sub>clip</sub>(T<sub>cand</sub>) is a number of subtrees in the translated code that have matched the subtrees of reference functions. This metric allows assessing the quality of a code translated in terms of its syntax.

* The translated code and reference code are also compared by their semantics, using data flows (```Guo et al., 2020```) when the source code is represented as a graph with its nodes being variables and its edges representing the ‘genetic’ relations between the nodes (denoting where the value of each variable comes from). The Match<sub>df</sub> metric shall be calculated according to the following formula:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{Match}_{df}&space;=&space;\frac{\textrm{Count}_{clip}(\textrm{DF}_{cand})}{\textrm{Count}(\textrm{DF}_{ref})},)

where Count(DF<sub>ref</sub>) is a total number of data flows in the reference code; Count<sub>clip</sub>(DF<sub>cand</sub>) is a number of data flows in the translated code that have matched the reference code.

## Subtask 2 - Handwritten Text Recognition

The key metric used to evaluate the participants’ solutions for this subtask is **String Accuracy** – the ratio of the number of completely matched transcriptions of strings to the number of all strings in the sample. It shall be calculated as follows:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\text{StringAcc}&space;=&space;\frac{\sum_{i=1}^n&space;[\text{pred}_i&space;=&space;\text{true}_i]}{n})

Here n is the size of the test sample, pred<sub>i</sub> is a string of characters that the model recognized on the i-th image in the sample, and true<sub>i</sub> is the correct translation of the i-th image produced by the annotator, [•] - Iverson bracket:

![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/Iverson.png)

The String Accuracy metric varies from 0 to 1, where 0 is the worst value and 1 is the best one.

## Subtask 3 - Zero-shot Object Detection

The quality will be evaluated using the **F1-score**:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{F1}=&space;2&space;\cdot&space;\frac{\text{Recall}\cdot\text{Precision}}{\text{Recall}&space;&plus;&space;\text{Precision}})

The F1-score is calculated based on Precision and Recall, which, in turn, depend on a set of prediction statistics - true positive (TP), false positive (FP) and false negative (FN):

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{Precision}=&space;\frac{\text{True\&space;Positive}}{\text{True\&space;Positive}&space;&plus;&space;\text{False\&space;Positive}},)

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{Recall}=&space;\frac{\text{True\&space;Positive}}{\text{True\&space;Positive}&space;&plus;&space;\text{False\&space;Negative}})

The rules according to which the prediction of the model belongs to one of the types are the following:

* if the given class from the request is absent in the correct annotations (that is, it is a negative example), but the participant's model made a prediction for it, the prediction is considered as *FP*
* if the given class from the request is present in the correct annotations (that is, it is a positive example):
  * if the participant's model did not make a prediction for it, that is, passed an empty list, or the number of predicted bboxes is smaller than the number of the correct ones for this class – the unpredicted bboxes are considered as *FN*
  * for each bbox of a given class from the prediction (a class can have multiple corresponding bboxes in the image):
    * if the intersection of the predicted bbox with at least one of the correct bboxes for this class by IoU > 0.5 - the prediction is considered as *TP*
    * if the intersection of the predicted bbox with each of the correct bboxes for the given label by IoU < 0.5 - the prediction is considered as *FP*.
    
IoU is a metric that evaluates the quality of the match between the predicted bbox and the reference one. It is calculated as the ratio of the intersection area to the area of the union of these two bboxes:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{IoU}&space;=&space;\frac{&space;\textrm{Intersection}}{&space;\textrm{Union}})

The IoU for each pair (prediction/true) takes a value from 0 to 1. The IoU cutoff threshold is 0.5, that is, all predicted bboxes with an IoU value less than 0.5 are considered as false predictions.

The F1-score metric varies from 0 to 1, where 0 is the worst value and 1 is the best one.

## Subtask 4 - Visual Question Answering

The quality will be evaluated using the **accuracy** metric. It reflects the percentage of correct matches for the pairs of predicted and correct answers, i.e. the percentage of matching answers (the model predicts an answer matching the true one) to the total number of answers. This metric varies from 0 to 1, where 0 is the worst value and 1, the best one:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{accuracy}&space;=\frac{&space;\textrm{True&space;answers}}{&space;\textrm{All&space;answers}})