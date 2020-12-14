import logging
import json
import os
import logging 
import sys
import re
import pandas as pd

__all__ = [
    "load_arrow_data",
    "load_label_list",
    "print_arrow_output",
    "read_report",
]

util_logger = logging.getLogger('transformer_tools.util.tagger_utils')


def load_label_list(labels):
    util_logger.info('Loading label list...')
    label_list = ["O"]
    # with open(labels) as my_labels:
    my_labels = [l.strip() for l in labels.split(";")]
    
    for line in my_labels:
        line = line.strip()
        label_list.append("%s" % line)

    out_list = list(set(label_list))
    util_logger.info('Output list: %s' % out_list)
    return out_list

_ARROWS = {
    "u"   : "B-up",
    "d"   : "B-down",
    "="   : "B-=",
    "up" : "B-up",
    "down" : "B-down",
    "↑"   : "B-up",
    "↓"   : "B-down",
}
    
_REVERSE_ARROWS = {
    "B-up" : "↑",
    "B-down" : "↓",
    "B-="    : "=",
}
    

def load_arrow_data(config,split):
    """Load the arrow data from json 

    :param data_dir: the target data directory 
    :param split: the target split 
    :raises: ValueError 
    :returns: a pandas `DataFrame`
    """
    data_dir = config.data_dir
    target_labels = set(load_label_list(config.label_list))
    
    target_file = os.path.join(data_dir,"%s.jsonl" % split)
    if not os.path.isfile(target_file):
        raise ValueError('Unknown file: %s' % target_file)

    util_logger.info('Reading: %s' % target_file)
    full_data = []
    with open(target_file) as my_target:
        for k,line in enumerate(my_target):
            json_line = json.loads(line.strip())
            text_input = json_line["question"]["stem"].split()
            output_tags = json_line["output"].split() 

            if len(text_input) != len(output_tags):
                util_logger.warning('Mismatched tags on line %d, skipping' % k)
                continue

            for m,word in enumerate(text_input):
                tag = "%s" % _ARROWS.get(output_tags[m],output_tags[m])
                if tag not in target_labels:
                    raise ValueError('Unknown label encountered: %s' % tag)
                full_data.append([k,word,tag])

    util_logger.info('Loading %d %s instances...' % (len(full_data),split))
    return pd.DataFrame(full_data,columns=["sentence_id", "words", "labels"])

def print_arrow_output(predictions,
                           eval_data,
                           split,
                           output_dir):
    """Print the arrow output 

    :param predictions: the model output predictions 
    :param eval_data: the evaluation data (in pandas format) 
    :param split: the particular split being tested
    :param output_dir: the output directory 
    """
    split_out = os.path.join(output_dir,"%s_output.tsv" % split)
    
    with open(split_out,'w') as my_output: 
        for example_id,predicted_labels in enumerate(predictions):
            original_sentence = [w for w in eval_data.loc[lambda df : df['sentence_id'] == example_id]["words"]]
            gold_labels = [w for w in eval_data.loc[lambda df : df['sentence_id'] == example_id]["labels"]]

            if len(gold_labels) != len(predicted_labels):
                util_logger.warning('Wrong output size!, skipping...')
                continue 
            elif len(gold_labels) != len(original_sentence):
                util_logger.warning('Wrong gold label size!, skipping...')
                continue

            ## compute an overlap score
            correct = len([t for n,t in enumerate(predicted_labels) if t == gold_labels[n]])
            score = correct/len(gold_labels)

            # ### print both outputs 
            print("%s\t%s\t%f" % (
                ' '.join(["%s%s" % (w,_REVERSE_ARROWS.get(gold_labels[k],gold_labels[k])) \
                              for k,w in enumerate(original_sentence)]),
                ' '.join(["%s%s" % (w,_REVERSE_ARROWS.get(predicted_labels[k],predicted_labels[k])) \
                              for k,w in enumerate(original_sentence)]),
                score
            ),file=my_output)


def read_report(output_dir):
    """Read the classification report 

    :param output_dir: the directory where it should sit
    """
    report_loc = os.path.join(output_dir,"eval_results.txt")
    label_scores = {}
    util_logger.info('Reading the classification report file...')
    
    if not os.path.isfile(report_loc):
        util_logger.info('Cannot find the output report...')
        return label_scores 

    with open(report_loc) as report:
        for line in report:
            fields = [l for l in re.split("\s+",line) if l.strip()]
            if len(fields) == 5: 
                label = fields[0]
                label_scores["%s_precision" % label] = float(fields[1])
                label_scores["%s_recall" % label] = float(fields[2])
                label_scores["%s_f1-score" % label] = float(fields[3])
                label_scores["%s_support" % label] = float(fields[4])

    util_logger.info(label_scores)
    return label_scores
