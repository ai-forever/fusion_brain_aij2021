# -*- coding: utf-8 -*-

import argparse
import json

from vqaEval import VQAEval


def eval(true_json, pred_json):

    quesIds = set(true_json.keys())

    assert quesIds == set(pred_json.keys()), 'Not all questions have predicted answers!'

    vqaEval = VQAEval(true_json, pred_json, n=3)
    vqaEval.evaluate(quesIds, true_json, pred_json)

    return vqaEval.accuracy['overall']


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path', type=str, default='./true_VQA.json',
                        help="the path to the directory with the gold answers")
    parser.add_argument('--pred_path', type=str, default='./prediction_VQA.json',
                        help="the path to the directory with the predictions")

    args = parser.parse_args()

    true_json = json.loads(open(args.ref_path, encoding="utf-8").read())
    pred_json = json.loads(open(args.pred_path, encoding="utf-8").read())

    acc = eval(true_json, pred_json)

    print(acc)
