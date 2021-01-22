from optparse import OptionParser,OptionGroup
from transformer_tools.Base import ConfigurableClass

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

    config.add_option_group(group)
