import sys
import os
import shutil
import subprocess
from optparse import OptionParser,OptionGroup
from transformer_tools import get_config
from transformer_tools import initialize_config

### general templates

## example input [python, -m, transformer_tools, Tagger, --output_dir, /output, --data_dir, /inputs, --num_train_epochs, "3", --learning_rate, "0.00001", --train_batch_size, "16", --seed, "42", --cloud, --early_stopping, --label_list, B-up;B-down;B-=, --dev_eval, --print_output, --tensorboard_dir, /output, --wandb_project, polarity_projection, --wandb_api_key, 68c5293beacfd1d762d7d4446d95852bd058efe1, --wandb_name, bert_base_polarity_projection]

_TAGGER_TEMPLATE = """description: %s
tasks:
- spec:
    image: %s
    resultPath: %s
    args: %s
    env:
      BATCH_SIZE: "16"
      DATASET: polarity_2
      LR: "0.00001"
      MAX_ITERATIONS: "3"
      MODEL: BERT
    datasetMounts:
    - datasetId: %s
      containerPath: /inputs
    requirements:
      gpuCount: 1
  cluster: %s
"""

## overall parameters


def params(config):
    """Main parameters for running the T5 model

    :param config: the global configuration object
    """
    group = OptionGroup(config,"transformer_tools.tools.run_beaker",
                            "Settings for quickly running beaker experiments")

    group.add_option("--experiment_type",
                     dest="experiment_type",
                     default='tagger',
                     type=str,
                     help="The type of experiment/utility to run [default='']")

    group.add_option("--image",
                     dest="image",
                     default='',
                     type=str,
                     help="The target image to use [default='im_i9s2ym270j3s']")

    group.add_option("--input_mount",
                     dest="input_mount",
                     default='',
                     type=str,
                     help="The path to the mounted input data [default='']")

    group.add_option("--cluster_id",
                     dest="cluster_id",
                     default='us_wvnghctl47k0/01E5TXXY2BN0XNQHF7QHSDJ5YR',
                     type=str,
                     help="The particular cluster to use [default='us_wvnghctl47k0/01E5TXXY2BN0XNQHF7QHSDJ5YR']")

    group.add_option("--beaker_group",
                     dest="beaker_group",
                     default='',
                     type=str,
                     help="The particular beaker group to assign to experiment (optional) [default='']")

    group.add_option("--beaker_command",
                     dest="beaker_command",
                     default='',
                     type=str,
                     help="The particular command to run [default='']")

    group.add_option("--result_path",
                     dest="result_path",
                     default='/output',
                     type=str,
                     help="The default output path for beaker [default='/output']")

    ##
    config.add_option_group(group)


def run_tagger_experiment(beaker_config,model_params):
    """Run a tagging experiment with beaker config and 
    model config 
    
    :param beaker_config: the beaker settings 
    :param config: the model configuration 
    """
    command = beaker_config.beaker_command.split()
    config = initialize_config(command,model_params)

from transformer_tools.Tagger import params as tparams

BEAKER_EX_TYPES = {
    "tagger" : (_TAGGER_TEMPLATE,tparams,run_tagger_experiment),
}

def main(argv):
    beaker_config = initialize_config(argv,params)

    ### get experiment type and template
    beaker_exp = BEAKER_EX_TYPES.get(beaker_config.experiment_type)
    if beaker_exp is None:
        raise ValueError('Unknown experiment type! %s' % beaker_config.experiment_type)
    if not beaker_config.beaker_command:
        raise ValueError('Must specify a beaker command!')

    template,model_params,run_func = beaker_exp
    #config = initialize_config(argv,model_params)

    ### run the beaker experiments 
    run_func(beaker_config,model_params)
