import logging
import json
import os
import logging 
import sys
import pandas as pd

__all__ = [
    "load_arrow_data",
    "load_label_list",
]

util_logger = logging.getLogger('transformer_tools.util.tagger_utils')


def load_label_list(labels):
    util_logger.info('Loading label list...')
    label_list = ["O"]
    # with open(labels) as my_labels:
    my_labels = [l.strip() for l in labels.split(";")]
    
    for line in my_labels:
        line = line.strip()
        label_list.append("B-%s" % line)

    out_list = list(set(label_list))
    util_logger.info('Output list: %s' % out_list)
    return out_list

_ARROWS = {
    "u" : "up",
    "d" : "down",
    "=" : "=",
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
                tag = "B-%s" % _ARROWS.get(output_tags[m],output_tags[m])
                if tag not in target_labels:
                    raise ValueError('Unknown label encountered: %s' % tag)
                full_data.append([k,word,tag])

    util_logger.info('Loading %d %s instances...' % (len(full_data),split))
    return pd.DataFrame(full_data,columns=["sentence_id", "words", "labels"])
