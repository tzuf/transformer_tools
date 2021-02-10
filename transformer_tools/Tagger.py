# #### an interface to the `simple_transformers` NER models
import json
import os
import logging 
import sys
import json
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax
from simpletransformers.ner import NERModel
from optparse import OptionParser,OptionGroup
from transformer_tools.util.tagger_utils import *
from transformer_tools import initialize_config
from transformer_tools.model import Model,init_wandb

## wandb (if available)
try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

util_logger = logging.getLogger('transformer_tools.Tagger')

def push_model(config):
    """Push models as artifacts 
    
    :param config: the global configuration 
    """
    run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name="",
    )
    util_logger.info('Backing up the model files to wandb')
    martifact = wandb.Artifact('%s_model' % config.wandb_name, type='model')

    for model_file in [
            "pytorch_model.bin",
            "special_tokens_map.json",
            "training_args.bin",
            "vocab.txt",
            "model_args.json",
            "config.json",
    ]:
        martifact.add_file(
                os.path.join(config.output_dir,model_file)
        )

    run.log_artifact(martifact)
    run.finish()
    
    
class TaggerModel(Model):
    """Base class for building tagger models

    """
    def __init__(self,model,config):
        self.model = model
        self.config = config

    @classmethod
    def from_config(cls,config):
        """Load tagger model from configuration 

        :param config: the global configuration instance 
        """
        pass

    def load_data(self,split='train'):
        """Load data for running experiments 

        :param split: the particular split to load 
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls,config):
        """Loads a model from configuration 

        :param config: the global configuration 
        """
        ## find labels in list
        label_list = load_label_list(config.label_list)
        use_cuda = True if torch.cuda.is_available() else False
        
        model = NERModel(
            config.model_name,
            config.model_type,
            use_cuda=use_cuda,
            labels=label_list,
            args={
                "fp16" : False,
                "classification_report" : True,
                "tensorboard_dir" : config.tensorboard_dir,
                "wandb_project" : config.wandb_project,
                "wandb_kwargs" : {
                    "name"    : config.wandb_name,
                    "entity"  : config.wandb_entity,
                    }
                }
        )
        return cls(model,config)

    def train_model(self):
        """Main method for training the data 

        :rtype: None 
        """
        self.logger.info('Loading the data...')
        train_data = self.load_data(split="train")
        dev_data = self.load_data(split="dev")

        self.logger.info('Training the model, outputdir=%s...' % self.config.output_dir)
        self.model.train_model(
            train_data,
            eval_data=dev_data,
            output_dir=self.config.output_dir,
            show_running_loss=False,
            args={
                "overwrite_output_dir" : True,
                "reprocess_input_data": True,
                "learning_rate"       : self.config.learning_rate,
                "num_train_epochs"    : self.config.num_train_epochs,
                "train_batch_size"    : self.config.train_batch_size,
                "eval_batch_size"     : self.config.eval_batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "use_early_stopping" : self.config.early_stopping,
                "fp16" : False,
                "classification_report" : True,
                "evaluate_during_training" : True,
                "evaluate_during_training_verbose" : True,
                "best_model_dir": self.config.output_dir,
                ## saving options (default only saves best model)
                "save_model_every_epoch" : self.config.save_model_every_epoch,
                "save_steps" : self.config.save_steps,
                "save_optimizer_and_scheduler" : self.config.save_optimizer_and_scheduler,
            })

    def eval_model(self,split='dev',print_output=False):
        """Evaluate the model

        :param split: the target split
        :param print_output: 
        """
        eval_data = self.load_data(split="dev")
        result, model_outputs, predictions = self.model.eval_model(
                eval_data,
                output_dir=self.config.output_dir,
        )
        if print_output:
            print_arrow_output(predictions,
                                   eval_data,
                                   split,
                                   self.config.output_dir)
            report_items = read_report(self.config.output_dir)

        result.update(report_items)
        return result

    def __enter__(self):
        self.logger.info('Entering the trainer...')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        try:
            self.logger.info('Closing wandb.')
            wandb.finish()
        except: pass

class ArrowTagger(TaggerModel):

    def load_data(self,split='train'):
        """Load data for running experiments 

        :param split: the particular split to load 
        """
        return load_arrow_data(self.config,split)

class GenericTagger(TaggerModel):
    """Generic data model 

    """
    pass

def params(config):
    """Main parameters for running the T5 model

    :param config: the global configuration object
    """
    from transformer_tools.model import params as mparams
    mparams(config)

    group = OptionGroup(config,"transformer_tools.Tagger",
                            "Settings for tagger models")

    group.add_option("--model_type",
                         dest="model_type",
                         default='bert-base-uncased',
                         type=str,
                         help="The type of tagger to use [default='bert-base-cased']")

    group.add_option("--model_name",
                         dest="model_name",
                         default='bert',
                         type=str,
                         help="The name of the model [default='bert']")

    group.add_option("--tagger_model",
                         dest="tagger_model",
                         default='arrow_tagger',
                         type=str,
                         help="The name of the model [default='arrow_tagger']")

    group.add_option("--label_list",
                         dest="label_list",
                         default='',
                         type=str,
                         help="The types of labels to use [default='']")

    group.add_option("--save_model_every_epoch",
                         dest="save_model_every_epoch",
                         action='store_true',
                         default=False,
                         help="Backup up every model after epoch [default=False]")

    group.add_option("--save_optimizer_and_scheduler",
                         dest="save_optimizer_and_scheduler",
                         action='store_true',
                         default=False,
                         help="Save the optimizer and schuler [default=False]")

    group.add_option("--save_steps",
                         dest="save_steps",
                         default=-1,
                         type=int,
                         help="Save model at this frequency [default=-1]")


    config.add_option_group(group)

_TAGGERS = {
    "arrow_tagger" : ArrowTagger,
}

def TaggerModel(config):
    """Factor for loading a tagger model 

    :param config: the global configuration 
    :raises: ValueError
    """
    tclass = _TAGGERS.get(config.tagger_model)
    if tclass is None:
        raise ValueError('Unknown tagger: %s' % config.tagger_model)
    if not config.label_list:
        raise ValueError('Must specify a label list!')
    
    return tclass.from_config(config)


def main(argv):
    """Main execution point 

    :param argv: the main cli arguments
    :rtype: None 
    """
    ## config
    config = initialize_config(argv,params)

    ## load wandb
    init_wandb(config) #add_name=True,add_entity=True)

    #model = TaggerModel(config)
    json_out = {}
    json_out["train_data"] = config.train_name
    json_out["eval_data"]  = config.eval_name

    if not config.no_training:
        with TaggerModel(config) as model: 
            model.train_model()

        ## save wandb model 
        if wandb_available and config.save_wandb_model:
            push_model(config)

    if config.dev_eval:
        dev_out = model.eval_model(
            split='dev',
            print_output=config.print_output
        )
        for key,value in dev_out.items():
            json_out["dev_%s" % key] = value
        
    if config.test_eval:
        test_out = model.eval_model(
            split='test',
            print_output=config.print_output
        )
        for key,value in dev_out.items():
            json_out["test_%s" % key] = value

    ### additional details 
    json_out["model_name"]  = config.model_name
    json_out["eval_name"]   = config.eval_name
    json_out["train_name"]  = config.train_name

    if json_out:
        metric_out = os.path.join(config.output_dir,"metrics.json")
        util_logger.info('Attempting to print metrics file: %s' % metric_out)
        with open(metric_out,'w') as my_metrics:
            my_metrics.write(
                json.dumps(json_out,indent=4)
            )

        #### log to wandb output 
        if wandb_available and config.wandb_project:
            wandb.log(json_out)

