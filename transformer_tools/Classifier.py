# #### an interface to the `simple_transformers` classification models
import json
import os
import logging 
import sys
import json
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax
from simpletransformers.classification import ClassificationModel
from optparse import OptionParser,OptionGroup
from transformer_tools import initialize_config
from transformer_tools.model import Model,init_wandb
from transformer_tools.util.classifier_utils import *
from optparse import Values

### should link in a separate model
from transformer_tools.Tagger import wandb_setup,push_model,load_wandb_data

# # Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
# train_data = [["Example sentence belonging to class 1", 1], ["Example sentence belonging to class 0", 0]]
# train_df = pd.DataFrame(train_data)
# eval_data = [["Example eval sentence belonging to class 1", 1], ["Example eval sentence belonging to class 0", 0]]
# eval_df = pd.DataFrame(eval_data)
# # Create a ClassificationModel
# model = ClassificationModel("roberta", "roberta-base")
# # Train the model
# model.train_model(train_df)
# # Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(eval_df)




class ClassifierModel(Model):

    def __init__(self,model,config):
        self.model = model
        self.config = config

    def load_data(self,split="train"):
        """Load data for running experiments 

        :param split: the particular split to load 
        """
        raise NotImplementedError

    def train_model(self):
        pass

class BinaryClassifier(ClassifierModel):

    @classmethod
    def from_config(cls,config):
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
            "max_seq_length" : config.max_seq_length,
        }

        ## load the classifier model
        model = ClassificationModel(
            config.model_name,
            config.model_type,
            use_cuda=use_cuda,
            args=global_args,
            num_labels=2,
        )

        return cls(model,config)


def params(config):
    """Main parameters for running the T5 model

    :param config: the global configuration object
    """
    from transformer_tools.model import params as mparams
    mparams(config)

    group = OptionGroup(config,"transformer_tools.Classifier",
                            "Settings for classifier models")

    group.add_option("--model_type",
                         dest="model_type",
                         default='allenai/longformer-base-4096',
                         type=str,
                         help="The type of tagger to use [default='longformer-base-4096']")

    group.add_option("--existing_model",
                         dest="existing_model",
                         default='',
                         type=str,
                         help="The path of an existing model to load [default='']")

    group.add_option("--model_name",
                         dest="model_name",
                         default='longformer',
                         type=str,
                         help="The name of the model [default='longformer']")

    group.add_option("--classifier_model",
                         dest="classifier_model",
                         default='binary',
                         type=str,
                         help="The name of classification model [default='binary']")

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

    group.add_option("--max_seq_length",
                         dest="max_seq_length",
                         default=128,
                         type=int,
                         help="The maximum sequence length [default=128]")    


    config.add_option_group(group)


_CLASSIFIERS = {
    "binary" : BinaryClassifier,
}

def ClassifierModel(config):
    """Factory method for loading classifier 
    
    :param config: the global classifier 
    """
    cclass = _CLASSIFIERS.get(config.classifier_model)
    if cclass is None:
        raise ValueError(
            "Unknown classifier model # types" 
        )
    # if config.wandb_model:
    #     return
    return cclass.from_config(config)
    

def main(argv):
    """Main execution point

    :param argv: the main cli arguments 
    :rtype: None 
    """
    ## config
    config = initialize_config(argv,params)

    ## load wandb data/models (if needed)
    if config.wandb_data or config.wandb_model: 
        wandb_setup(config)
        
    model = ClassifierModel(config)
