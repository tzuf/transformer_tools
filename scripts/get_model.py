""" 
Download trained T5 model to target directory
   
Usage:
    get_model.py [--source_file=<source_file> --save_dir=<save_dir>]

                         
Options:
   -h --help     Show this screen.
   --source_file=<source_file>    URL of source zip file containing model. [default: https://drive.google.com/uc?id=102f1PPY6nyO9k9EaQ6ZH6wJdH_VNGL-3]
   --save_dir=<save_dir>    Directory to save model to.  [default: saved_model]

"""

# T5 concat 1-13: https://drive.google.com/file/d/102f1PPY6nyO9k9EaQ6ZH6wJdH_VNGL-3/view?usp=sharing

from docopt import docopt
from pathlib import Path
import logging
import os
import gdown
import zipfile
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]

MODELS_DIR = ROOT / "_models"

if __name__ == "__main__":
    args = docopt(__doc__, version='0.1')
    save_dir = MODELS_DIR / args["--save_dir"]
    url = args['--source_file']
    output = 'data.zip'
    
    # create save dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading file to {str(save_dir)}!")
    gdown.download(url, output, quiet=False)
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(str(save_dir))
    
    # remove downloaded zip file
    os.remove(output)