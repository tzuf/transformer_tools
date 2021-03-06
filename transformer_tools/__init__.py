import os
import sys
import logging
import imp
from optparse import OptionParser,OptionGroup
from transformer_tools.util.loader import load_module as load_module
from transformer_tools.util.os_util import make_experiment_directory as make_wdir


USAGE = """usage: python -m transformer_tools mode [options] [--help]"""
DESCRIPTION = """Set of utilities for building language fragments and running experiments on them"""

_CONFIG = OptionParser(usage=USAGE,description=DESCRIPTION,conflict_handler="resolve")

## logging

_CONFIG.add_option("--logging",dest="logging",default='info',type=str,
                      help="The logging level [default='']")

_CONFIG.add_option("--log_file",dest="log_file",default='pipeline.log',
                      help="The name of the log file (if logging to file) [default='pipeline.log']")

_CONFIG.add_option("--override",dest="override",action='store_true',default=False,
                      help="Override the current working directory and creat it again [default=False]")

_CONFIG.add_option("--cloud",dest="cloud",action='store_true',default=False,
                      help="Called when used in cloud environment [default=False]")

_CONFIG.add_option("--wdir",dest="wdir",default='',
                      help="The specific working directory to set up [default='']")

_CONFIG.add_option("--cuda_device",dest="cuda_device",default=-1,type=int,
                      help="The cuda device to run on (for GPU processes) [default=-1]")

gen_config = _CONFIG


_LEVELS = {
    "info"  : logging.INFO,
    "debug" : logging.DEBUG,
    "warning" : logging.WARNING,
    "error"   : logging.ERROR,
    "quiet"   : logging.ERROR,
}

def _logging(config):
  """Basic logging settings 

  :param config: the global configuration 
  """
  level = _LEVELS.get(config.logging,logging.INFO)
  if config.wdir and config.log_file and config.log_file != "None":
    log_out = os.path.join(config.wdir,config.log_file)
    logging.basicConfig(filename=log_out,level=level)

    ## redirect stdout to wdir (e.g., all of the tqdm stuff) 
    sys.stdout = open(os.path.join(config.wdir,"stdout.log"),'w')
    sys.stderr = open(os.path.join(config.wdir,"stderr.log"),'w')

  else:
    logging.basicConfig(level=level)

def initialize_config(argv,params=None):
    """Create a config and set up the global logging
    
    :param argv: the cli input 
    :param params: the additional parameters to add 
    """
    if params: params(_CONFIG)
    config,_ = _CONFIG.parse_args(argv)

    if config.wdir:
        wdir = make_wdir(config.wdir,config=config)
    _logging(config)
    return config    

def _load_module(module_path):
    """load a particular zubr module using format:
    zubr.module1.module12.ect.. 

    :param module_path: path of module to be loaded
    :type module_path: str
    :returns: loaded module
    :rtype: module 
    """
    try: 
        mod = __import__(module_path,level=0)
        for c in module_path.split('.')[1:]:
            mod = getattr(mod,c)
        return mod    
    except Exception as e:
        raise e
        #util_logger.error(e,exc_info=True)

_SHORTCUTS = {
    # "t5_classifier"  : "transformer_tools.solver",
    # "t5_generator"   : "transformer_tools.lang",
}

def get_config(module_name,logging='info'):
    """Return back a configuration instance for a utility with default values 

    :param module: the name of the module to use
    """
    mod = _load_module(_SHORTCUTS.get(module_name,module_name))
    if hasattr(mod,"params"):
        config = initialize_config(["--logging",logging],mod.params)
        return config
    raise ValueError('No config for this module' % module_name)

## factories
#from transformer_tools.T5Generative import T5GenerativeFactory as T5Generator
from transformer_tools.T5Generative import LoadModel as LoadT5Generator
#from transformer_tools.T5Classification import T5ClassificationModel as T5Classifier
from transformer_tools.T5Classification import LoadModel as LoadT5Classifier


### wandb stuff
#from transformer_tools.util.wandb_util import load_wandb
