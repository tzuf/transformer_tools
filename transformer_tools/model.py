from optparse import OptionParser,OptionGroup
from transformer_tools.Base import ConfigurableClass

class Model(ConfigurableClass):
    """Base model for all other models
    """
    def query(self,text_input):
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


    # group.add_option("--cuda_device",dest="cuda_device",default=-1,type=int,
    #                   help="The cuda device to run on (for GPU processes) [default=-1]")

    ## TODO : move the t5base settings here 

    config.add_option_group(group)
