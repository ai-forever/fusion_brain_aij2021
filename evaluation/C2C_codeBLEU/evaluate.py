# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

# -*- coding:utf-8 -*-
import argparse
import bleu
import weighted_ngram_match
import syntax_match
import dataflow_match
import json

parser = argparse.ArgumentParser()
parser.add_argument('--ref_path', type=str, required=True,
                        help='reference file')
parser.add_argument('--pred_path', type=str, required=True, 
                        help='prediction file')
parser.add_argument('--lang', type=str, default='python', 
                        choices=['python'],
                        help='programming language')
parser.add_argument('--params', type=str, default='0.25,0.25,0.25,0.25',
                        help='alpha, beta and gamma')
parser.add_argument('--verbose', default=False, action='store_true')

args = parser.parse_args()

lang = args.lang
alpha,beta,gamma,theta = [float(x) for x in args.params.split(',')]

# preprocess inputs

with open(args.ref_path, 'r') as f1:
    ref = json.load(f1)

with open(args.pred_path, 'r') as f2:
    pred = json.load(f2)

references = []
predictions = []

for key in ref.keys():
    references.append([ref[key]])
    try:
        predictions.append(pred[key])
    except KeyError:
        print(f"No predictions for the index {key}!")
        break

# calculate ngram match (BLEU)
tokenized_preds = [x.split() for x in predictions]
tokenized_refs = [[x.split() for x in reference] for reference in references]

ngram_match_score = bleu.corpus_bleu(tokenized_refs,tokenized_preds)


# calculate weighted ngram match
keywords = [x.strip() for x in open('keywords/'+args.lang+'.txt', 'r', encoding='utf-8').readlines()]
def make_weights(reference_tokens, key_word_list):
    return {token:1 if token in key_word_list else 0.2 \
            for token in reference_tokens}

tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)]\
            for reference_tokens in reference] for reference in tokenized_refs]

weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights,tokenized_preds)

# calculate syntax match
syntax_match_score = syntax_match.corpus_syntax_match(references, predictions, args.lang)

# calculate dataflow match
dataflow_match_score = dataflow_match.corpus_dataflow_match(references, predictions, args.lang)

code_bleu_score = alpha*ngram_match_score\
                + beta*weighted_ngram_match_score\
                + gamma*syntax_match_score\
                + theta*dataflow_match_score

if args.verbose:
    print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'.\
                    format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))

print('CodeBLEU score: ', code_bleu_score)
