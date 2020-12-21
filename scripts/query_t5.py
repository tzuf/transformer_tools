""" 
Query trained T5 model
   
Usage:
    query_t5.py QUERY [--model_dir=<model_dir>]

                         
Options:
   -h --help     Show this screen.
   --model_dir=<model_dir>    Directory to save model to.  [default: _models/saved_model/concat_1_13]

"""

# T5 concat 1-13: https://drive.google.com/file/d/102f1PPY6nyO9k9EaQ6ZH6wJdH_VNGL-3/view?usp=sharing

from docopt import docopt
from pathlib import Path
import logging
import os
import gdown
import zipfile
import sys

ROOT = Path(__file__).parents[1]
sys.path.append(str(ROOT))
from transformer_tools import LoadT5Classifier,get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




if __name__ == "__main__":
    args = docopt(__doc__, version='0.1')
    model_dir = args["--model_dir"]
    query = args["QUERY"]
    print(model_dir)
    logger.info("Loading model...")
    gen_config = get_config("transformer_tools.T5Generative")
    gen_config.target_model = model_dir
    gen_config.max_answer = 200
    model = LoadT5Generator(gen_config)
    res = model.query(query)
    logger.info(f"[Result]: {res}")