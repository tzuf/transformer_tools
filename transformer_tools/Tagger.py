# #### an interface to the `simple_transformers` NER models
import json
import os
import logging 
import sys
import torch
import numpy as np
import pandas as pd
from scipy.special import softmax
from simpletransformers.ner import NERModel
from optparse import OptionParser,OptionGroup
from transformer_tools.util.tagger_utils import *

from transformer_tools.Base import (
    ConfigurableClass,
    UtilityClass,
    LoggableClass,
)

util_logger = logging.getLogger('transformer_tools.Tagger')
    

class TaggerModel(ConfigurableClass):
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

        self.logger.info('Training the model...')
        self.model.train_model(
            train_data,
            eval_data=dev_data,
            output_dir=self.config.output_dir,
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
            })

        if self.config.dev_eval:
            result, model_outputs, predictions = self.model.eval_model(
                dev_data,
                output_dir=self.config.output_dir,
            )

class ArrowTagger(TaggerModel):

    def load_data(self,split='train'):
        """Load data for running experiments 

        :param split: the particular split to load 
        """
        return load_arrow_data(self.config,split)
    

def params(config):
    """Main parameters for running the T5 model

    :param config: the global configuration object
    """
    from transformer_tools.T5Base import params as tparams
    tparams(config)

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

    group = OptionGroup(config,"transformer_tools.NER",
                            "Settings for NER transformer models")

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
    from transformer_tools import initialize_config
    config = initialize_config(argv,params)

    model = TaggerModel(config)
    if not config.no_training: 
        model.train_model()

if __name__ == "__main__":
    main(sys.argv[1:])


# # Creating train_df  and eval_df for demonstration
# train_data = [
#     [0, "Simple", "B-MISC"],
#     [0, "Transformers", "I-MISC"],
#     [0, "started", "O"],
#     [0, "with", "O"],
#     [0, "text", "O"],
#     [0, "classification", "B-MISC"],
#     [1, "Simple", "B-MISC"],
#     [1, "Transformers", "I-MISC"],
#     [1, "can", "O"],
#     [1, "now", "O"],
#     [1, "perform", "O"],
#     [1, "NER", "B-MISC"],
# ]
# train_df = pd.DataFrame(train_data, columns=["sentence_id", "words", "labels"])

# eval_data = [
#     [0, "Simple", "B-MISC"],
#     [0, "Transformers", "I-MISC"],
#     [0, "was", "O"],
#     [0, "built", "O"],
#     [0, "for", "O"],
#     [0, "text", "O"],
#     [0, "classification", "B-MISC"],
#     [1, "Simple", "B-MISC"],
#     [1, "Transformers", "I-MISC"],
#     [1, "then", "O"],
#     [1, "expanded", "O"],
#     [1, "to", "O"],
#     [1, "perform", "O"],
#     [1, "NER", "B-MISC"],
# ]
# eval_df = pd.DataFrame(eval_data, columns=["sentence_id", "words", "labels"])

# # Create a NERModel
# model = NERModel("bert", "bert-base-cased", args={"overwrite_output_dir": True, "reprocess_input_data": True})
# print(model)

# # # Train the model
# # model.train_model(train_df)

# # # Evaluate the model
# # result, model_outputs, predictions = model.eval_model(eval_df)


# # Predictions on arbitary text strings
# # sentences = ["Some arbitary sentence", "Simple Transformers sentence"]
# # predictions, raw_outputs = model.predict(sentences)

# # print(predictions)

# # # More detailed preditctions
# # for n, (preds, outs) in enumerate(zip(predictions, raw_outputs)):
# #     print("\n___________________________")
# #     print("Sentence: ", sentences[n])
# #     for pred, out in zip(preds, outs):
# #         key = list(pred.keys())[0]
# #         new_out = out[key]
# #         preds = list(softmax(np.mean(new_out, axis=0)))
# #         print(key, pred[key], preds[np.argmax(preds)], preds)
