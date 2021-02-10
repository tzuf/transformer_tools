import os
import sys
import logging
from optparse import OptionParser,OptionGroup
from transformer_tools.Base import ConfigurableClass
try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

util_logger = logging.getLogger('transformer_tools.model')

## wandb boilerplate

def init_wandb(config,add_name=False,add_entity=False):
    """Initializes the overall wandb environment 

    :param config: the global configuration 
    :raises: ValueError 
    """
    if "WANDB_NOTES" not in os.environ and config.wandb_note:
        os.environ["WANDB_NOTES"] = config.wandb_note
    if "WANDB_API_KEY" not in os.environ:
        if not config.wandb_api_key:
            raise ValueError(
                'Unknown wandb key! please let environment variable or specify via `--wandb_api_key`'
            )
        util_logger.info('Setting the wandb api key....')
        os.environ["WANDB_API_KEY"] = config.wandb_api_key
    ## hide the key if provided
    # if add_name:
    #     os.environ["WANDB_NAME"] = config.wandb_name
    # if add_entity:
    #     os.environ["WANDB_ENTITY"] = config.wandb_entity
        
    config.wandb_api_key = None

class Model(ConfigurableClass):
    """Base model for all other models
    """
    def query(self,text_input,prefix='answer:'):
        """Main method for outside interaction with Python/text 

        :param text_input: the input text 
        :rtype text_input: str 
        :param prefix: the model mode to run (if needed) 
        :rtype: obj
        """
        raise NotImplementedError

def params(config):
    """Main parameters for running the T5 model

    :param config: the global configuration object
    """
    group = OptionGroup(config,"transformer_tools.model",
                            "Generic settings for models")

    ### wandb stuff
    
    group.add_option("--wandb_project",
                         dest="wandb_project",
                         default=None,
                         help="The particular wandb project (if used) [default='']")

    group.add_option("--wandb_cache",
                         dest="wandb_cache",
                         default='~/.wandb_cache',
                         help="The particular wandb project (if used) [default='']")

    group.add_option("--wandb_api_key",
                         dest="wandb_api_key",
                         default='',
                         type=str,
                         help="The particular wandb api key to use [default='']")

    group.add_option("--wandb_name",
                         dest="wandb_name",
                         default='new experiment (default)',
                         type=str,
                         help="The particular wandb api key to use [default='new experiment (default)']")
    
    group.add_option("--wandb_note",
                         dest="wandb_note",
                         default='empty',
                         type=str,
                         help="The note to use for the wandb [default='empty']")

    group.add_option("--wandb_model",
                         dest="wandb_model",
                         default='',
                         type=str,
                         help="Specifies a location to an existing wandb model [default='']")

    group.add_option("--tensorboard_dir",
                         dest="tensorboard_dir",
                         default=None,
                         help="The types of labels to use [default=None]")

    group.add_option("--train_name",
                         dest="train_name",
                         default="generic",
                         type=str,
                         help="The name of training data [default='generic']")

    group.add_option("--eval_name",
                         dest="eval_name",
                         default="generic",
                         type=str,
                         help="The name of evaluation data [default='generic']")

    group.add_option("--save_wandb_model",
                         dest="save_wandb_model",
                         action='store_true',
                         default=False,
                         help="Backup the wandb model [default=False]")

    group.add_option("--wandb_entity",
                         dest="wandb_entity",
                         default='',
                         type=str,
                         help="Backup the wandb model [default='']")

    group.add_option("--model_name",
                         dest="model_name",
                         default='n/a',
                         type=str,
                         help="The type of model (for plotting purposes) [default='n/a']")

    ## TODO : move the t5base settings here

    group.add_option("--gradient_accumulation_steps",
                         dest="gradient_accumulation_steps",
                         default=1,
                         type=int,
                         help="number of gradient accumulations [default=1]")

    group.add_option("--seed",
                         dest="seed",
                         default=42,
                         type=int,
                         help="random seed[default=42]")

    group.add_option("--weight_decay",
                         dest="weight_decay",
                         default=0.0,
                         type=float,
                         help="the weight decay amount [default=0.0]")

    group.add_option("--adam_epsilon",
                         dest="adam_epsilon",
                         default=1e-8,
                         type=float,
                         help="adam epsilon parameter [default=1e-8]")

    group.add_option("--warmup_steps",
                         dest="warmup_steps",
                         default=0,
                         type=int,
                         help="warmnup steps [default=0]")

    group.add_option("--num_train_epochs",
                         dest="num_train_epochs",
                         default=3,
                         type=int,
                         help="number of training iterations [default=3]")

    group.add_option("--no_shuffle",
                         dest="no_shuffle",
                         action='store_true',
                         default=False,
                         help="Remove shuffling [default=False]")

    group.add_option("--learning_rate",
                         dest="learning_rate",
                         default=3e-4,
                         type=float,
                         help="learning rate [default=3e-5]")

    group.add_option("--train_batch_size",
                         dest="train_batch_size",
                         default=16,
                         type=int,
                         help="batch size [default=3]")

    group.add_option("--remove_models",
                         dest="remove_models",
                         action='store_true',
                         default=False,
                         help="Remove models/checkpoints [default=False]")

    group.add_option("--remove_checkpoints",
                         dest="remove_checkpoints",
                         action='store_true',
                         default=False,
                         help="Remove models/checkpoints [default=False]")

    group.add_option("--adafactor",
                         dest="adafactor",
                         action='store_true',
                         default=False,
                         help="Use adafactor [default=False]")

    group.add_option("--print_output",
                         dest="print_output",
                         action='store_true',
                         default=False,
                         help="Print output [default=False]")

    group.add_option("--no_training",
                         dest="no_training",
                         action='store_true',
                         default=False,
                         help="Skip the training step [default=False]")

    group.add_option("--dev_eval",
                         dest="dev_eval",
                         action='store_true',
                         default=False,
                         help="run an evaluation of the dev eval [default=False]")

    group.add_option("--train_eval",
                         dest="train_eval",
                         action='store_true',
                         default=False,
                         help="run an evaluation of the train eval [default=False]")

    group.add_option("--test_eval",
                         dest="test_eval",
                         action='store_true',
                         default=False,
                         help="run an evaluation of the test [default=False]")

    group.add_option("--early_stopping",
                         dest="early_stopping",
                         action='store_true',
                         default=False,
                         help="Use early stopping [default=False]")

    group.add_option("--patience",
                         dest="patience",
                         default=5,
                         type=int,
                         help="Patient level (when early stopping is used) [default=5]")

    group.add_option("--n_gpu",
                         dest="n_gpu",
                         default=1,
                         type=int,
                         help="The number of gpus to use [default=1]")

    group.add_option("--max_grad_norm",
                         dest="max_grad_norm",
                         default=1.0,
                         type=float,
                         help="maximum gradient norm [default=1.0]")

    group.add_option("--data_dir",
                         dest="data_dir",
                         default='',
                         type=str,
                         help="The directory where the data sits [default='']")

    group.add_option("--wandb_data",
                         dest="wandb_data",
                         default='',
                         type=str,
                         help="Link to the wandb data [default='']")
    
    group.add_option("--eval_batch_size",
                         dest="eval_batch_size",
                         default=8,
                         type=int,
                         help="the size of the eval batch size [default=8]")
    
    group.add_option("--model_name_or_path",
                         dest="model_name_or_path",
                         default='t5-base',
                         help="The type of dataset to train [default='']")
    
    group.add_option("--model_dir",
                         dest="model_dir",
                         default='',
                         help="The model to use for eval [default='']")
    
    group.add_option("--tokenizer_name_or_path",
                         dest="tokenizer_name_or_path",
                         default='t5-base',
                         help="The type of dataset to train [default='']")

    group.add_option("--output_dir",
                         dest="output_dir",
                         default='',
                         help="The location to put output for model [default='']")

    group.add_option("--fp_16",
                         dest="fp_16",
                         action='store_true',
                         default=False,
                         help="use fp_16 precision [default=False]")

    ## tpu cores
    group.add_option("--tpu_cores",
                         dest="tpu_cores",
                         default=0,
                         type=int,
                         help="The number of TPU cores (for tpu usage) [default=0]")

    group.add_option("--special_device",
                         dest="special_device",
                         default='cuda',
                         type=str,
                         help="The special device (for loading) [default='cuda']")
    ## loading models and checkpoints

    group.add_option("--checkpoint_path",
                         dest="checkpoint_path",
                         default='',
                         type=str,
                         help="Path to checkpoint (for loading model) [default=T5Classification]")
    
    group.add_option("--target_model",
                         dest="target_model",
                         default='',
                         type=str,
                         help="Path to target model (for loading model) [default=T5Classification]")

    group.add_option("--verbose",
                         dest="verbose",
                         action='store_true',
                         default=False,
                         help="Verbose option [default=False]")

    group.add_option("--auto_lr_find",
                         dest="auto_lr_find",
                         action='store_true',
                         default=False,
                         help="automatic learning rate finder [default=False]")

    group.add_option("--data_subdir",
                         dest="data_subdir",
                         default='',
                         type=str,
                         help="The subdirectory to find the data (if needed) [default='']")

    config.add_option_group(group)
