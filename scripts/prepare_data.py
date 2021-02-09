""" 
Convert data dir of files in bAbI format to T5 format, and add as artifact to folder in wandb
   
Usage:
    prepare_data.py [--source_dir=<source_dir> --save_name=<save_name> --wandb --just_push_wandb]

                         
Options:
   -h --help     Show this screen.
   --source_dir=<source_dir>    Directory containing data in bAbI format. [default: .]
   --save_name=<save_name>    Directory to save data to (in '_data' dir).  [default: default]
   --just_push_wandb   If True, skip data-preprocessing and just push `save_name` folder to wandb. [default: False]
   --wandb   If True, save also as artifact on wandb.

"""


from docopt import docopt
from pathlib import Path
import logging
import os
import sys
import subprocess

import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[1]
DATA_DIR = ROOT / "_data"
CONVERT_SCRIPT = ROOT / "bin" / "babi2json.py"
PY_BIN = sys.executable


if __name__ == "__main__":
    args = docopt(__doc__, version='0.1')
    
    source_dir = Path(args["--source_dir"])
    save_name = DATA_DIR / args["--save_name"]
    just_wandb = args["--just_push_wandb"]
    add_to_wandb = args["--wandb"] or just_wandb
    
    # create save dir
    save_name.mkdir(parents=True, exist_ok=True)
    
    
    # check if we want to skip preprocessing and just push some folder to wandb
    if not just_wandb:
        logger.info(f"Preparing data at {str(save_name)}...")
        
        cmd = [PY_BIN, str(CONVERT_SCRIPT), "--data_loc", str(source_dir), "--odir", str(save_name)]
        print(' '.join(cmd))
        try:
            res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
        
    if add_to_wandb:
        wand_save_path = str(save_name.relative_to(ROOT))
        logger.info(f"Pushing data to wandb from {wand_save_path}...")
        run = wandb.init(job_type='eval', project="t5_data", entity="eco-semantics")
        artifact = wandb.Artifact(save_name.name, type='dataset')
        artifact.add_dir(str(save_name))
        run.log_artifact(artifact)
    