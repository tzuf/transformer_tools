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
from optparse import Values

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
    util_logger.info('Backing up the model files to wandb')
    martifact = wandb.Artifact('%s_model' % config.wandb_name, type='model')
    martifact.add_dir(os.path.join(config.output_dir,"best_model"))
    #matrifact.add_file(os.path.join(config.output_dir,"trainer_config.json"))
    wandb.log_artifact(martifact)

def wandb_setup(config):
    """Set up the wandb data and models if such are specified 

    :param config: the global configuration 
    :rtype: None 
    """
    init_wandb(config)
    run = wandb.init(entity=config.wandb_entity)
    
    if config.wandb_data:
        run = wandb.init(entity=config.wandb_entity)
        artifact = run.use_artifact(config.wandb_data, type='dataset')
        artifact_dir = artifact.download()
        util_logger.info('Download data to: %s' % artifact_dir)
        adir = os.path.join(artifact_dir,config.data_subdir) if config.data_subdir else artifact_dir
        config.data_dir = adir

    if config.wandb_model:
        model = run.use_artifact(config.wandb_model, type='model')
        model_dir = model.download()
        util_logger.info('Download data to: %s' % model_dir)
        config.existing_model = model_dir
        config.wandb_model = ""

    run.finish()
    
def load_wandb_data(config):
    """Load the wandb data and also set `subdir` path if needed 

    :param config: the global configuration 
    """
    run = wandb.init(entity=config.wandb_entity)
    artifact = run.use_artifact(config.wandb_data, type='dataset')
    artifact_dir = artifact.download()
    util_logger.info('Download data to: %s' % artifact_dir)
    adir = os.path.join(artifact_dir,config.data_subdir) if config.data_subdir else artifact_dir
    config.data_dir = adir

    
class TaggerModel(Model):
    """Base class for building tagger models

    """
    def __init__(self,model,config):
        self.model = model
        self.config = config


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

        global_args = {
            "fp16" : False,
            "classification_report" : True,
            "tensorboard_dir" : config.tensorboard_dir,
            "wandb_project" : config.wandb_project,
            "wandb_kwargs" : {
                "name"    : config.wandb_name,
                "entity"  : config.wandb_entity,
                }
            }

        model = NERModel(
            config.model_name,
            config.model_type,
            use_cuda=use_cuda,
            labels=label_list,
            args=global_args,
        )
        return cls(model,config)

    @classmethod
    def load_existing(cls,config,silent=True):
        """Load an existing model from configuration 

        :param config: the global configuration
        """
        use_cuda = True if torch.cuda.is_available() else False
        if config.wandb_model:
            wandb_setup(config)

        ## load original configuration
        orig_config = None 
        with open(os.path.join(config.existing_model,"trainer_config.json")) as oconfig:
            orig_config = Values(json.loads(oconfig.read()))
        orig_config.existing_model = config.existing_model

        model = NERModel(
            orig_config.model_name,
            orig_config.existing_model,
            use_cuda=use_cuda,
            args={"silent" : silent},
        )
        return cls(model,orig_config)

    def train_model(self):
        """Main method for training the data 

        :rtype: None 
        """
        self.logger.info('Loading the data...')
        train_data = self.load_data(split="train")
        dev_data = self.load_data(split="dev")
        self.config.best_model = os.path.join(self.config.output_dir,"best_model")
        self.logger.info('Training the model, outputdir=%s...,best_model=%s' % (self.config.output_dir,self.config.best_model))

        train_params = {
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
            "best_model_dir": self.config.best_model,
            "save_model_every_epoch" : self.config.save_model_every_epoch,
            "save_steps" : self.config.save_steps,
            "save_optimizer_and_scheduler" : self.config.save_optimizer_and_scheduler,
            "save_best_model": True,
        }

        ## train the model 
        self.model.train_model(
            train_data,
            eval_data=dev_data,
            output_dir=self.config.output_dir,
            show_running_loss=False,
            args=train_params,
        )

        ## backing up the config and create pointer to best model 
        with open(os.path.join(self.config.best_model,"trainer_config.json"),'w') as mconfig:
            mconfig.write(json.dumps(self.config.__dict__))
        self.config.existing_model = self.config.best_model
        
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

    def query(self,text_input,prefix='answer:',convert_to_string=True):
        """Main method for outside interaction with Python/text 

        :param text_input: the input text 
        :rtype text_input: str 
        :param prefix: the model mode to run (if needed) 
        :rtype: obj
        """
        predictions, raw_outputs = self.model.predict([text_input])
        raw_outputs = [np.max(softmax([s[1][0] for s in v.items()][0])) for v in raw_outputs[0]]
        preds = [[(i[0],i[1],raw_outputs[k]) for i in p.items()][0] for k,p in enumerate(predictions[0])]
        return self._post_process_output(preds,convert_to_string=convert_to_string)

    def _post_process_output(self,predictions,convert_to_string):
        """Maps the predictions into a more comfortable tuple format

        :param predictions: the predictions made by the model 
        :type predictions: list 
        :rtype: list 
        """
        if convert_to_string:
            return ' '.join(["%s-%s" % (p[0],p[1]) for p in predictions])
        return predictions

class ArrowTagger(TaggerModel):

    def load_data(self,split='train'):
        """Load data for running experiments 

        :param split: the particular split to load 
        """
        return load_arrow_data(self.config,split)

    def _post_process_output(self,predictions,convert_to_string):
        """Maps the predictions into the arrows (if `convert_to_string=True`,
        which is the default) 

        :param predictions: the predictions made by the model 
        :type predictions: list 
        :rtype: list 
        """
        normalized = [(p[0],REVERSE_ARROWS.get(p[1],p[1]),p[2]) for p in predictions]
        if convert_to_string:
            return ' '.join(["%s%s" % (p[0],p[1]) for p in normalized])
        return normalized
    
class GenericTagger(TaggerModel):
    """Generic data model 

    """
    def load_data(self,split='train'):
        """Load data for running experiments 

        :param split: the particular split to load 
        """
        raise ValueError('Please implement me!')

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

    group.add_option("--existing_model",
                         dest="existing_model",
                         default='',
                         type=str,
                         help="The path of an existing model to load [default='']")

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
                         default="B-up;B-down;B-=",
                         type=str,
                         help="The types of labels to use [default='B-up;B-down;B-=']")

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
    if not config.label_list and not config.wandb_model:
        raise ValueError('Must specify a label list!')

    if config.wandb_model:
        return tclass.load_existing(config)
    return tclass.from_config(config)


def main(argv):
    """Main execution point 

    :param argv: the main cli arguments
    :rtype: None 
    """
    ## config
    config = initialize_config(argv,params)

    ## load wandb data/models (if needed) 
    wandb_setup(config) 

    model = TaggerModel(config)
    json_out = {}
    json_out["train_data"] = config.train_name
    json_out["eval_data"]  = config.eval_name

    if not config.no_training:
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
