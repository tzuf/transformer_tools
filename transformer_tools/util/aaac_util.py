### utility for T5-aaac
import re
import os
import sys
import json
import copy
import logging
import numpy as np
import pandas as pd
import string
import random
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction



util_logger = logging.getLogger('transformer_tools.util.t5_util')

def aaac_eval(output:pd.DataFrame=None,target:pd.DataFrame=None,mode_sequence=[]):
    output_keys = [m['to'] for m in mode_sequence]

    scores = None

    # SCORE final output
    last_key = output_keys[-1]
    ## score generated argdown reconstruction
    if last_key=='argdown_reconstruction':
        scores = evaluate_argdown(output=output[last_key], target=target[last_key])
    ## score generated reason statements
    if last_key=='argdown_reconstruction':
        scores = evaluate_reason(output=output[last_key], target=target[last_key])
    ## score generated conclusion statements
    if last_key=='argdown_reconstruction':
        scores = evaluate_conclusions(output=output[last_key], target=target[last_key])

    # Evaluate mutual consistency of generated reasons, conclusion statement and argdown
    rca_keys = ['reason_statements','conclusion_statements','argdown_reconstruction']
    if all(k in output_keys for k in rca_keys):
        scores = pd.concat([scores,evaluate_consistency_rca(output=output[rca_keys], target=target[rca_keys])],axis=1)

    return scores


def evaluate_argdown(output:pd.Series=None, target:pd.Series=None):
    data = pd.DataFrame([output,target],columns=['output','target'])
    def eval_ad_pair(pair):
        target=parse_argdown(pair['target']) 
        output=parse_argdown(pair['output']) 
        scores = {
            'argdown_valid_syntax':1 if output else 0,
            'argdown_used_prem_exist':used_prem_exist(output),
            'argdown_all_stat_used':prem_non_used(output),
            'argdown_used_prem_schemes':used_prem_schemes(output),            
            'argdown_bleu_global':bleu([pair['output']],[pair['target']],smoothing_function=SmoothingFunction().method1),
            'argdown_bleu_best_match':bleu_ad_best_match(output=output,target=target),
            'argdown_bleu_final_concl':bleu([output[-1]['text']],[target[-1]['text']],smoothing_function=SmoothingFunction().method1) if output else 0, # TODO CHECK 'text'
            'argdown_diff_num_prem':0 # TODO
        }
        return scores
    scores = pd.DataFrame(data.apply(eval_ad_pair,axis=1))
    return scores



def parse_argdown(ad_string):
    argdown = [{"text"="premise"}]
    return argdown 

def used_prem_exist(argdown):
    return 1

def prem_non_used(argdown):
    return 1

def used_prem_schemes(argdown):
    return 1

def bleu_ad_best_match(output=None,target=None):
    return 1


def evaluate_reason(output:pd.Series=None, target:pd.Series=None):
    pass


def evaluate_conclusions(output:pd.Series=None, target:pd.Series=None):
    pass


def evaluate_consistency_rca(output:pd.DataFrame=None, target:pd.DataFrame=None):
    pass


