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

    group.add_option("--wandb_project",
                         dest="wandb_project",
                         default=None,
                         help="The particular wandb project (if used) [default='']")

    group = OptionGroup(config,"transformer_tools.model",
                            "Generic settings for models")

    config.add_option_group(group)
