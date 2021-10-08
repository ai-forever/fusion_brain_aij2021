# -*- coding: utf-8 -*-
import argparse
import json


# String Accuracy calculation
def evaluate(pred_path, true_path):

    with open(true_path) as f1:
        true_data = json.load(f1)
    with open(pred_path) as f2:
        pred_data = json.load(f2)

    # [pred == true] / num_total
    numStringOK = 0
    numStringTotal = 0

    for key in true_data:
        true = true_data[key]
        if key in pred_data:
            pred = pred_data[key]
        else:
            pred = ''
            
        numStringOK += 1 if true == pred else 0
        numStringTotal += 1

    stringAccuracy = numStringOK / numStringTotal

    return stringAccuracy
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path', type=str, default='true_HTR.json', help="the path to the file with the gold answers")
    parser.add_argument('--pred_path', type=str, default='prediction_HTR.json', help="the path to the file with the predictions")

    args = parser.parse_args()
    
    stringAccuracy = evaluate(args.pred_path, args.ref_path)

    print(stringAccuracy)
