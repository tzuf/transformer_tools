import sys
import os
import subprocess
from optparse import OptionParser,OptionGroup

### general templates

_TAGGER_TEMPLATE = """description: %s
tasks:
- spec:
    image: %s
    resultPath: /output
    args: [python, -m, transformer_tools, Tagger, --output_dir, /output, --data_dir, /inputs, --num_train_epochs, "3", --learning_rate, "0.00001", --train_batch_size, "16", --seed, "42", --cloud, --early_stopping, --label_list, B-up;B-down;B-=, --dev_eval, --print_output, --tensorboard_dir, /output, --wandb_project, polarity_projection, --wandb_api_key, 68c5293beacfd1d762d7d4446d95852bd058efe1, --wandb_name, bert_base_polarity_projection]
    env:
      BATCH_SIZE: "16"
      DATASET: polarity_2
      LR: "0.00001"
      MAX_ITERATIONS: "3"
      MODEL: BERT
    datasetMounts:
    - datasetId: ds_ez2215eexjt2
      containerPath: /inputs
    requirements:
      gpuCount: 1
  cluster: us_wvnghctl47k0/01E5TXXY2BN0XNQHF7QHSDJ5YR
"""



def params(config):
    """Main parameters for running the T5 model

    :param config: the global configuration object
    """
    # from transformer_tools.T5Base import params as tparams
    # tparams(config)

    from transformer_tools.Tagger import params as mparams
    mparams(config)
    

    group = OptionGroup(config,"transformer_tools.tools.run_beaker",
                            "Settings for quickly running beaker experiments")

    group.add_option("--experiment_type",
                         dest="experiment_type",
                         default='',
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


    config.add_option_group(group)

def run_tagger_experiment(config):
    pass

BEAKER_EX_TYPES = {
    "tagger" : _TAGGER_TEMPLATE,
}

def main(argv):
    from transformer_tools import initialize_config
    config = initialize_config(argv,params)

    ##
    

    

