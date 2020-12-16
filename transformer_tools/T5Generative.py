import os
import time
import logging
import re
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from dataclasses import dataclass
from transformer_tools import initialize_config
from optparse import OptionParser,OptionGroup
from transformer_tools.util.t5_util import *

from transformer_tools.T5Base import (
    Text2TextData,
    T5LoggingCallback,
    T5Trainer,
    T5Text2TextBase,
    run_trainer_tester,
)

util_logger = logging.getLogger('transformer_tools.T5Generative')

_GENERATIVE_BUILDERS={
    "json_generative"  : generative_data,
}

class GenerativeText2Text(Text2TextData):
    """Text2Text data class specialized for classification

    :param DATA_BUILDERS: the type of data builders available for the 
      the classifier.
    """
    DATA_BUILDERS = _GENERATIVE_BUILDERS

class T5Generative(T5Text2TextBase):
    """Simple generative T5 Model
    
    :param EVALUATOR: the classifier evaluation code 
    :param LOADER: the classifier data loader class 
    """
    EVALUATOR = single_token_eval
    LOADER    = GenerativeText2Text

## lightning module

class T5GenerationBase(T5Text2TextBase):
    """Class for Pretrained ConditionalGeneration T5 Models 
    that specifically supports classification. 

    :param EVALUATOR: the classifier evaluation code 
    :param LOADER: the classifier data loader class 
    """
    #EVALUATOR = single_token_eval
    LOADER    = GenerativeText2Text

class T5GenenerationModel(T5GenerationBase):
    """This class implements explanations alongside a simple classifier

    NOTE : this is a specialized class tailored to my current Jsonl/GlossKB 
    file format. 

    :param EVALUATOR: the classifier evaluation code 
    :param LOADER: the classifier data loader class 
    """
    @torch.no_grad()
    def evaluate_output(self,dtype='dev',final_eval=False):
        """Method for evaluating output, called after training step (passes by default). Loads the 
        data manually so it provides more control over printing model output.
        
        :param final_eval: is the final evaluation?
        :rtype final_eval: bool
        :rtype: pytorch.Tensor
        """
        self.model_logger.info('Attempting to evaluate input with dtype=%s, final_eval=%s,device=%s' %\
                                   (dtype,final_eval,self._device))
                                   
        ## loads the dataset on each round, can't seem to find the trainer dataset anywhere!
        dataset = self._get_data(dtype,final_eval=final_eval)
        loader = DataLoader(dataset,
                                batch_size=self.hparams.eval_batch_size,
                                shuffle=False,
                                num_workers=self.hparams.num_workers)

        outputs = []; targets = []
        ofile = None if (not final_eval or not dataset.data_rep or not self.hparams.output_dir) else \
          os.path.join(self.hparams.output_dir,"%s_eval.tsv" % dtype)

        ## run the
        ## check mode
        gen_func = self._classification_step if not final_eval else self._generative_step
        output_size = self.hparams.max_answer if final_eval else self.hparams.classifier_length

        self.model_logger.info('Going through batches, ofile=%s, lenght of data rep=%d, func=%s, output_size=%d' %\
                                   (ofile,len(dataset.data_rep),gen_func.__name__,output_size))
                                   
        ## go through the batches 
        for batch in tqdm(loader):
            outs = gen_func(batch,max_length=output_size,
                                no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
                                num_beams=self.hparams.num_beams,
                                do_sample=self.hparams.do_sample,
                                top_p=self.hparams.top_p,
                                min_length=self.hparams.min_length,
                                top_k=self.hparams.top_k)

            dec    = [self.tokenizer.decode(ids.detach().cpu()) if self.tokenizer.decode(ids).strip() else "" for ids in outs]
            target = [self.tokenizer.decode(ids.detach().cpu()) for ids in batch["target_ids"].detach()]
            outputs.extend(dec)
            targets.extend(target)

        ### pass to custom evaluator to make sense of it
        score = bleu_eval(outputs,targets)

        #if ofile: print_full_output(outputs,targets,dataset.data_rep,ofile)
        if ofile: print_full_with_bleu(outputs,targets,dataset.data_rep,ofile)
        self.model_logger.info('Processed and scored %d outputs/targets, score=%f' % (len(outputs),score))
        ## 
        return score

class T5SeqGenerationModel(T5GenerationBase):
    """T5 generation model that is used (at eval time) for
    doing sequence prediction or sequence tagging (e.g., NER, polarity
    arrow prediction, etc..)
    """

    
    # TODO : merge with implementation in class above
    
    @torch.no_grad()
    def evaluate_output(self,dtype='dev',final_eval=False):
        """Method for evaluating output, called after training step (passes by default). Loads the 
        data manually so it provides more control over printing model output.
        
        :param final_eval: is the final evaluation?
        :rtype final_eval: bool
        :rtype: pytorch.Tensor
        """
        self.model_logger.info('Attempting to evaluate input with dtype=%s, final_eval=%s,device=%s' %\
                                   (dtype,final_eval,self._device))
                                   
        ## loads the dataset on each round, can't seem to find the trainer dataset anywhere!
        dataset = self._get_data(dtype,final_eval=final_eval)
        original_sizes = dataset.data_sizes

        loader = DataLoader(dataset,
                                #batch_size=self.hparams.eval_batch_size,
                                batch_size=1, ## set to 1, to make life easier
                                shuffle=False,
                                num_workers=self.hparams.num_workers)

        outputs = []; targets = []
        ofile = None if (not final_eval or not dataset.data_rep or not self.hparams.output_dir) else \
          os.path.join(self.hparams.output_dir,"%s_eval.tsv" % dtype)

        ## run the
        ## check mode
        gen_func = self._classification_step if not final_eval else self._generative_step
        output_size = self.hparams.max_answer if final_eval else self.hparams.classifier_length

        self.model_logger.info('Going through batches, ofile=%s, lenght of data rep=%d, func=%s, output_size=%d' %\
                                   (ofile,len(dataset.data_rep),gen_func.__name__,output_size))

        ## go through the batches
        curr = 0
        for batch in tqdm(loader):
            target_len = original_sizes[curr]
            curr += 1
            
            outs = gen_func(batch,max_length=target_len,#max_length=output_size,
                                no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
                                num_beams=self.hparams.num_beams,
                                do_sample=self.hparams.do_sample,
                                top_p=self.hparams.top_p,
                                #min_length=self.hparams.min_length,
                                min_length=target_len,
                                top_k=self.hparams.top_k)

            dec    = [self.tokenizer.decode(ids.detach().cpu()) if self.tokenizer.decode(ids).strip() else "" for ids in outs]
            target = [self.tokenizer.decode(ids.detach().cpu()) for ids in batch["target_ids"].detach()]
            outputs.extend(dec)
            targets.extend(target)

        ### pass to custom evaluator to make sense of it
        #score = bleu_eval(outputs,targets)

        #if ofile: print_full_output(outputs,targets,dataset.data_rep,ofile)
        #if ofile: print_full_with_bleu(outputs,targets,dataset.data_rep,ofile)
        if ofile: seq_eval(outputs,targets,dataset.data_rep,ofile)
        self.model_logger.info('Processed and scored %d outputs/targets, score=%f' % (len(outputs),score))
        ##
        return score

_GENERATION_MODELS={
    "T5GenerationModel"  : T5GenenerationModel,
    "T5SeqGeneration"    : T5SeqGenerationModel,
}
    
class T5GenerativeTrainer(T5Trainer):
    """Special trainer for T5 classification models


    :param _MODELS: Possible classifier models visible to the Classifier 
        trainer. 
    """
    _MODELS = _GENERATION_MODELS

def T5GenerativeFactory(config):
    """Factory method for loading a T5 model 

    :param config: the configuration 
    :returns: T5 model instance (loaded from configuration) 
    """
    mtype = _GENERATION_MODELS.get(config.T5_type,None)
    if mtype is None:
        raise ValueError('Unknown T5 Model...%s' % config.T5_type)
    return mtype
    

def params(config):
    """Main parameters for running the T5 model

    :param config: the global configuration object
    """
    from transformer_tools.T5Base import params as tparams
    tparams(config)

    group = OptionGroup(config,"transformer_tools.T5Classification",
                            "Settings for T5Classification models")

    group.add_option("--callback_monitor",
                         dest="callback_monitor",
                         default="val_loss",
                         type=str,
                         help="batch size [default='val_loss']")

    group.add_option("--T5_type",
                         dest="T5_type",
                         default='T5GenerationModel',
                         type=str,
                         help="The type of T5 model to use [default=T5GenerativeModel]")

    group.add_option("--gen_eval",
                         dest="gen_eval",
                         default='bleu',
                         type=str,
                         help="The type of generation evaluation to use [default='bleu']")

    group.add_option("--data_builder",
                         dest="data_builder",
                         default='json_generative',
                         type=str,
                         help="Dataset builder function [default='json_mcqa']")

    config.add_option_group(group)

def main(argv):
    """Main execution loop

    :param argv: the cli input 
    """
    config = initialize_config(argv,params)
    t5_class = T5GenerativeFactory(config)

    ### 
    run_trainer_tester(config,T5GenerativeTrainer,t5_class,
                           {"best_dev_score" : "dev_loss","dev_eval" : "dev_bleu"})
