import logging
import json
import os
import logging 
import sys
import re
import pandas as pd

__all__ = [
    "load_classification_data",
]

util_logger = logging.getLogger('transformer_tools.util.tagger_utils')


def load_classification_data(config,split):
    """Load classification data 

    :param config: the global configuration 
    :param split: the target split 
    """
    data_dir = config.data_dir
    target_file = os.path.join(data_dir,"%s.jsonl" % split)
    if not os.path.isfile(target_file):
        raise ValueError('Unknown file: %s' % target_file)
    
    util_logger.info('Reading: %s' % target_file)
    instances = []
    with open(target_file) as my_target:
        for k,line in enumerate(my_target):
            line = line.strip()
            json_line = json.loads(line)
            text_passage = json_line["question"]["stem"]
            output = json_line["output"]
            instances += [text_passage,output]
