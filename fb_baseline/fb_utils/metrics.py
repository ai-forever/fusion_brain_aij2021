import sklearn
import argparse
import json
import numpy as np
import itertools

import re
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py).


class VQAEval:
    def __init__(self, vqa, vqaRes, n=2):
        self.n = n
        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.vqa = vqa
        self.vqaRes = vqaRes
        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                             "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                             "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                             "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                             "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                             "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                             "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                             "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                             "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                             "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                             "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                             "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                             "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                             "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                             "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                             "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                             "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                             "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                             "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                             "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                             "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                             "youll": "you'll", "youre": "you're", "youve": "you've"}
        self.manualMap = {'none': '0',
                              'zero': '0',
                              'one': '1',
                              'two': '2',
                              'three': '3',
                              'four': '4',
                              'five': '5',
                              'six': '6',
                              'seven': '7',
                              'eight': '8',
                              'nine': '9',
                              'ten': '10'
                            }
        self.manualMap_ru = {'ноль': '0',
                             'нисколько': '0',
                             'никакой': '0',
                             'один': '1',
                             'два': '2',
                             'три': '3',
                             'четыре': '4',
                             'пять': '5',
                             'шесть': '6',
                             'семь': '7',
                             'восемь': '8',
                             'девять': '9',
                             'десять': '10'
                            }
        self.articles = ['a', 'an', 'the']
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!']

    def evaluate(self, quesIds, true_json, pred_json):

        true_res = true_json
        pred_res = pred_json

        accQA = []

        for quesId in quesIds:

            lang = true_res[quesId]['lang']
            resAns = pred_res[quesId]
            resAns = resAns.replace('\n', ' ')
            resAns = resAns.replace('\t', ' ')
            resAns = resAns.strip()
            true_acc = []
            true_ans = true_res[quesId]['answer']

            resAns = self.processPunctuation(resAns)
            resAns = self.processDigitArticle(resAns, lang)

            matchingAns = []
            if resAns == true_ans:
                matchingAns.append(resAns)

            acc = min(1, float(len(matchingAns)))
            true_acc.append(acc)

            avgGTAcc = float(sum(true_acc))/len(true_acc)
            accQA.append(avgGTAcc)

            self.setEvalQA(quesId, avgGTAcc)

        self.setAccuracy(accQA)

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText, lang):
        outText = []
        tempText = inText.lower().split()

        if lang == 'ru':
            for word in tempText:
                word_parsed = morph.parse(word)[0]
                lemma = word_parsed.normal_form
                lemma = self.manualMap_ru.setdefault(lemma, lemma)
                outText.append(lemma)
        else:
            for word in tempText:
                word = self.manualMap.setdefault(word, word)
                if word not in self.articles:
                    outText.append(word)
                else:
                    pass
            for wordId, word in enumerate(outText):
                if word in self.contractions:
                    outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    def setAccuracy(self, accQA):
        self.accuracy['overall'] = round(100*float(sum(accQA))/len(accQA), self.n)

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100*acc, self.n)


def levenshtein_distance(first, second):
    distance = [[0 for _ in range(len(second) + 1)] for _ in range(len(first) + 1)]
    for i in range(len(first) + 1):
        for j in range(len(second) + 1):
            if i == 0:
                distance[i][j] = j
            elif j == 0:
                distance[i][j] = i
            else:
                diag = distance[i - 1][j - 1] + (first[i - 1] != second[j - 1])
                upper = distance[i - 1][j] + 1
                left = distance[i][j - 1] + 1
                distance[i][j] = min(diag, upper, left)
    return distance[len(first)][len(second)]


def cer(pred_texts, gt_texts):
    assert len(pred_texts) == len(gt_texts)
    lev_distances, num_gt_chars = 0, 0
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        lev_distances += levenshtein_distance(pred_text, gt_text)
        num_gt_chars += len(gt_text)
    return lev_distances / num_gt_chars


def wer(pred_texts, gt_texts):
    assert len(pred_texts) == len(gt_texts)
    lev_distances, num_gt_words = 0, 0
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        gt_words, pred_words = gt_text.split(), pred_text.split()
        lev_distances += levenshtein_distance(pred_words, gt_words)
        num_gt_words += len(gt_words)
    return lev_distances / num_gt_words


def string_accuracy(pred_texts, gt_texts):
    assert len(pred_texts) == len(gt_texts)
    correct = 0
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        correct += int(pred_text == gt_text)
    return correct / len(gt_texts)


def acc(targets, outputs):
    targets = targets.data.cpu().numpy()
    outputs = outputs.data.cpu().numpy().argmax(axis=1)
    return sklearn.metrics.accuracy_score(targets, outputs)


## Detection ##
def xywh_to_xyxy(bbox):
    '''Из стандартного coco формата [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
    '''
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def iou(box1, box2):
    '''Считает IoU между двумя bbox, каждый box [xmin, ymin, xmax, ymax].
    Args:
      box1: (list) bbox, sized (4,).
      box2: (list) bbox, sized (4,).
    Return:
      float: iou
    '''
    lt = np.zeros(2)
    rb = np.zeros(2)    # get inter-area left_top/right_bottom
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
    wh[wh < 0] = 0    # if no overlapping 
    inter = wh[0] * wh[1]
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])  
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter / (area1 + area2 - inter)  
    if np.isnan(iou):
        iou = 0
    return iou


def detection_evaluate(true_annot, predictions):
    '''Основная функция для подсчета метрик.
       Файл с правильными ответами имеет следующую структуру (это словарь) : {img_name1 : {label1:[[xmin,ymin,w,h],[xmin,ymin,w,h]], label2:[[xmin,ymin,w,h]]}, ....}
    
       0. создаем пустые переменные для статистик TP,FP,FN.
       1. Пробегаемся по каждому изображению из true (ключи true - имена файлов изображений).
       2. Внутри каждого изображения пробгеаемся по всем лейблам (ключи true[img_name] - названия лейблов).
       3. true = true_annot[img][label] - правильные боксы для данного лейбла на данном изображении. соответственно
          pred = predictions[img][label] - предсказанные боксы для данного лейбла на данном изображении.
       4. если len(true)==0 (то есть это был негативный лейбл в запросах), но модель участника сделала предсказание для него,
       увеличиваем FP на один за каждое предсказание
       5. если len(true)>0, но модель участника не сделала ни одного предсказания, то
       увеличиваем FN за каждый не найденные правильный бокс
       6. если len(true)>0 и если len(pred)>0:
       
       для начала смотрим пересечение по IoU true боксов со всеми pred боксами. Допустим предсказаний для изображения было 3, правильных бокса - 2. Тогда для каждого правильного бокса смотрим пересечение по IoU со всеми предсказанными, и если нет пересечений выше >0.5 ни с одним боксом, то мы считаем это за FN для каждого true бокса
       
       Затем мы пробегаемся по всем предсказаниям. Если у данного предсказания нет пересечения по IoU выше 0.5 ни с одним правильным боксом, мы считаем это предсказание за FP. Если же есть хоть с одним - считаем за TP.
       7. Затем считаются метрики precision и recall, и на их основе - финальный F1_score.
    
    
    Args:
      ref_path: (string) path to pred.json
      pred_path: (string) path to true.json
    Return:
      float: f1_score
    '''
    assert list(true_annot.keys()) == list(predictions.keys()), 'Not all images have predictions!'
        
    fp = 0
    fn = 0
    tp = 0

    for img in true_annot:
        for label in true_annot[img]:
            true = true_annot[img][label]
            assert label in predictions[img], f'There are no prediction for label "{label}" for image {img} in requests!'

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
                
                pairs = list(itertools.product(true,pred))
                pairs_iou = [(el[0],el[1],iou(xywh_to_xyxy(el[0]),xywh_to_xyxy(el[1]))) for el in pairs]
                
                for _, group in itertools.groupby(pairs_iou,key=lambda x: x[0]):#true
                    if np.all(np.array([i for _, _, i in group]) < 0.5):
                        fn += 1
        
                
                for _, group in itertools.groupby(sorted(pairs_iou,key=lambda x: x[1]),key = lambda x: x[1]):#pred
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
############


## VQA ##
def vqa_evaluate(vqa_result):
    true_json, pred_json = {}, {}
    for i, row in vqa_result.iterrows():
        true_json[str(i)] = {
            "answer": row['gt_output'],
            "lang": "en"
        }
        pred_json[str(i)] = row['pred_output']
    quesIds = list(true_json.keys())

    assert quesIds == list(pred_json.keys()), 'The order of predictions doesn’t match the order of targets!'

    vqaEval = VQAEval(true_json, pred_json, n=2)
    vqaEval.evaluate(quesIds, true_json, pred_json)

    return vqaEval.accuracy['overall']
############
