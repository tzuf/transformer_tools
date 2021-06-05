## base classes for T5 
import argparse
from argparse import ArgumentParser
import h5py
import os
import json
import time
import logging
import random
import re
import sys
import numpy as np
import ntpath
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from tqdm.auto import tqdm
from sklearn import metrics as sklearn_metrics
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from transformers import PreTrainedTokenizer
from transformer_tools.model import Model,init_wandb,WANDB_CACHE

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

from transformers.optimization import ( 
    Adafactor
)

from optparse import OptionParser,OptionGroup

from transformer_tools.Base import (
    ConfigurableClass,
    UtilityClass,
    LoggableClass,
)

from transformer_tools.util.t5_util import *

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False
    
util_logger = logging.getLogger('transformer_tools.T5Model')

## T5 modeling, based largely on notebook from
### https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb

## TODO: replace method below with something more sensible

def _update_config(args,config):
    """Update the settings of the configuration 

    :param args: the new config inside of the model (from original model version)
    :param config: config for this experiment 
    :rtype: None 
    """
    ### data settings 
    args.data_dir   = config.data_dir
    args.output_dir = config.output_dir
    args.wdir       = config.wdir
    args.checkpoint_path = config.checkpoint_path
    args.seed = config.seed

    ### decoder options
    args.num_beams            = config.num_beams
    args.do_sample            = config.do_sample
    args.no_repeat_ngram_size = config.no_repeat_ngram_size
    args.top_p         = float(config.top_p) if config.top_p is not None else config.top_p
    args.top_k         = int(config.top_k) if config.top_k is not None else config.top_k
    args.min_length    = int(config.min_length) if config.min_length is not None else config.min_length
    args.regen_k       = int(config.regen_k) if config.regen_k is not None else config.regen_k

    ###
    args.early_stop_decoding = config.early_stop_decoding
    args.max_seq_length = config.max_seq_length
    args.max_answer = config.max_answer
    args.classifier_length = config.classifier_length

    ## batch
    args.train_batch_size = config.train_batch_size
    args.eval_batch_size = config.eval_batch_size
    args.print_output = config.print_output
    args.verbose = config.verbose
    ## training
    args.learning_rate = config.learning_rate
    #args.train_batch_size = config.train_batch_size
    args.no_shuffle = config.no_shuffle
    args.num_train_epochs = config.num_train_epochs
    args.remove_models = config.remove_models
    args.remove_checkpoints = config.remove_checkpoints
    args.callback_monitor  = config.callback_monitor
    args.weight_decay = config.weight_decay
    args.adam_epsilon = config.adam_epsilon
    args.gradient_accumulation_steps = config.gradient_accumulation_steps
    args.dev_eval = config.dev_eval
    args.test_eval = config.test_eval
    args.train_eval = config.train_eval

    ## data builder 
    args.data_builder = config.data_builder
    args.retrain_batch = config.retrain_batch
    args.print_bleu    = config.print_bleu
    args.generate_once = config.generate_once
    args.no_generate_train = config.no_generate_train
    args.wandb_project = config.wandb_project
    args.wandb_name = config.wandb_name
    args.wandb_note = config.wandb_note
    args.save_wandb_model = config.save_wandb_model

    try:
        args.max_regenerate = config.max_regenerate
        args.generation_prefix = config.generation_prefix
    except:
        pass 
    

class Text2TextData(LoggableClass):
    """Base class for Dataset objects

    """
    DATA_BUILDERS = {}

    def __init__(self,inputs,
                     targets,
                     data_rep=[],
                     data_sizes=[],
                     ):
        """Creates a Text2TextData instance
        
        :param inputs: the model inputs (encoder side)
        :param targets: the model output (decoder side)
        :param evaluator: the evaluation function to use 
        :param data_rep: the original data representation (if needed) 
        """
        self.inputs    = inputs
        self.targets   = targets
        self.data_rep  = data_rep
        self.data_sizes =data_sizes
        self.logger.info('Loaded Text2TextData instance with #inputs=%d,#targets=%d' %\
                             (len(self.inputs),len(self.targets)))

    def __getitem__(self,index):
        ## grabbing items 
        source_ids  = self.inputs[index]["input_ids"].squeeze()
        target_ids  = self.targets[index]["input_ids"].squeeze()
        src_mask    = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids,
                "source_mask": src_mask,
                "target_ids": target_ids,
                "target_mask": target_mask}

    def __len__(self):
        return len(self.inputs)

    @classmethod
    def from_file(cls,config,
                      tokenizer,
                      split,
                      final_eval):
        """Load data from file 

        :param config: the global configuration 
        :param tokenizer: the model tokenizer 
        :param full_generation: full generation 
        :raises: ValueError 
        """
        ### find dataset class first
        builder = cls.DATA_BUILDERS.get(config.data_builder,None)
        if builder is None: raise ValueError('Unknown data builder=%s' % config.data_builder)
        
        ## build data using builder
        o = builder(config,tokenizer,split,final_eval)
        if len(o) == 3: 
            inputs,outputs,data_rep = o #builder(config,tokenizer,split,final_eval)
            data_sizes = [] 
        else:
            inputs,outputs,data_rep,data_sizes = o
        ### 
        return cls(inputs,outputs,data_rep=data_rep,
                       data_sizes=data_sizes)
    
### T5 Callbacks 

class T5LoggingCallback(pl.Callback,LoggableClass):
    """Logger for T5"""

    def on_validation_end(self,trainer,pl_module):
        """Called on validation end 

        :param trainer: the main trainer object
        :param pl_module: the lightning module running the model
        """
        self.logger.info("Validation Results...")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    self.logger.info("{} = {}".format(key.strip(), str(metrics[key]).strip()))

    def on_test_end(self,trainer,pl_module):
        """Called after testing 
        
        :param trainer: the trainer object 
        :param pl_module: the lightning module running the model
        """
        self.logger.info("Testing Results...")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    value = metrics[key]
                    if torch.is_tensor(value) or (hasattr(value, 'data') and torch.is_tensor(value.data)):
                        value = value.detach().cpu()
                    self.logger.info("{} = {}".format(key.strip(), str(value).strip()))

    def on_init_start(self,trainer):
        self.logger.info('Starting the checkpoint')

    def on_init_end(self,trainer):
        self.logger.info('Finished setting up the logger callback')



def _recursive_detach(in_dict):
    """Detach all tensors in `in_dict`.
    May operate recursively if some of the values in `in_dict` are dictionaries
    which contain instances of `torch.Tensor`. Other types in `in_dict` are
    not affected by this utility function.
    Parameters
    ----------
    in_dict : dict
    Returns
    -------
    out_dict : dict
    """
    out_dict = {}
    for k, v in in_dict.items():
        if isinstance(v, dict):
            out_dict.update({k: _recursive_detach(v)})
        elif callable(getattr(v, 'detach', None)):
            out_dict.update({k: v.detach().cpu()})
        else:
            out_dict.update({k: v})
    return out_dict        
        

class T5Text2TextBase(pl.LightningModule):
    """Base model for T5 fine-tuning model"""
    LOADER = Text2TextData
    EVALUATOR = None


    def __init__(self,hparams,tokenizer=None,model_loc=None):
        """Creates a T5 FineTuner instance 

        :param hparams: the model hyper-parameters 
        :param model: the underlying T5 model 
        :param tokenizer: the model tokenizer
        :param dclass: the data class 
        :param evaluator: the evaluator function 
        """
        super(T5Text2TextBase, self).__init__()
        self.hparams   = hparams
        if model_loc is not None: self._model = T5ForConditionalGeneration.from_pretrained(model_loc)
        else: self._model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        self.dclass     = type(self).LOADER
        self.evaluator  = type(self).EVALUATOR
        ## tokenizer 
        if tokenizer is None: self._tokenizer = T5Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)
        else: self._tokenizer = T5Tokenizer.from_pretrained(tokenizer)
        self.model_logger.info('Loaded T5 Tuner model...')

    ## prepare data (backups tokenizer and parameters)

    def prepare_data(self):
        """This method checks that provided data is an the right format
        for extracting explanations; it is a little hacky at this moment, and
        relies on a utility function. 
        
        :rtype: None 
        """
        self.model_logger.info('Backing up the tokenizer')
        self._backup_tokenizer()

    def regenerate_eval(self,split='dev'):
        """Regenerate just for evaluation 
    
        :param split: the target split to regenerate
        """
        self.model_logger.warning('Regenerating data, nothing to do here...')

    def _backup_tokenizer(self):
        self.tokenizer.save_pretrained(self.hparams.output_dir)
        with open(os.path.join(self.hparams.output_dir,'commandline_args.txt'), 'w') as f:
            json.dump(dict(self.hparams.items()), f, indent=2)

    ## save protocol (to include modified tokenizer)
    ## see: https://github.com/PyTorchLightning/pytorch-lightning/issues/1755

    def on_save_checkpoint(self,checkpoint):
        checkpoint["state_dict"] = None 
        self._model.save_pretrained(self.hparams.output_dir)

    @classmethod
    def load(cls,checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        tokenizer  = checkpoint['tokenizer']
        hparams    = checkpoint['hparams']
        
        ## assumed model is stored where the checkpoint is 
        model_path = os.path.dirname(checkpoint_path)
        model = cls(hparams,tokenizer=tokenizer,model_loc=model_path)
        return model

    def change_model(self,model_path):
        """Switch model (basically a model setter)

        :param model_path: the path to the model to change to
        """
        previous_device = self._model.device
        self.model_logger.info('Changing model to %s, previous device=%s' % (model_path,previous_device))
        self._model = T5ForConditionalGeneration.from_pretrained(model_path)
        self._model.to(previous_device)
        self.model_logger.info('new device=%s' % self._model.device)

    ### OPTIMIZER
    
    def configure_optimizers(self):
        """Setup the main optimizer

        :returns: the main optimizer 
        """
        model = self._model
        no_decay = ["bias", "LayerNorm.weight"]
        parameters_first = [p for n,p in model.named_parameters() if \
                            not any(nd in n for nd in no_decay)]
        parameters_sec = [p for n,p in model.named_parameters() if \
                              any(nd in n for nd in no_decay)]

        optimizer_grouped_parameters = [
            {"params": parameters_first,"weight_decay" : self.hparams.weight_decay},
            {"params" : parameters_sec,"weight_decay": 0.0}
        ]

        if self.hparams.adafactor:
            optimizer = Adafactor(optimizer_grouped_parameters,
                                      lr=self.hparams.learning_rate,
                                      relative_step=False
                                      )
        else: 
            optimizer = AdamW(optimizer_grouped_parameters,
                                  lr=self.hparams.learning_rate,
                                  eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch,
                           batch_idx,
                           optimizer,
                           optimizer_idx,
                           second_order_closure=None,
                           using_native_amp=None):
        """Runs the optimizer step 
        """
        if self.trainer.use_tpu: xm.optimizer_step(optimizer)
        else: optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    ### GENERAL STEPS 

    def _step(self,batch):
        """Runs a single batch

        :param batch: the batch to run 
        :returns: the model loss 
        """
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(input_ids=batch["source_ids"],
                           attention_mask=batch["source_mask"],
                           lm_labels=lm_labels, # change to labels
                           #labels=lm_labels,
                           decoder_attention_mask=batch['target_mask'])

        loss = outputs[0]
        return loss

    def forward(self,input_ids,
                    attention_mask=None,
                    decoder_input_ids=None,
                    decoder_attention_mask=None,
                    lm_labels=None):
        """Main forward function for model 

        :param input_ids: the input ids 
        :param attention_mask: the masks for the input
        """
        return self.model(input_ids,
                              attention_mask=attention_mask,
                              decoder_input_ids=decoder_input_ids,
                              decoder_attention_mask=decoder_attention_mask,
                              labels=lm_labels)
                              #lm_labels=lm_labels)

    ## VALIDATION STEPS AND EPOCHS
    
    def validation_step(self, batch, batch_idx):
        """Runs a single step over the validation data 

        :param batch: the target batch 
        :param batch_idx: the path id 
        :rtype: dict
        :returns: dictionary that includes loss 
        """
        loss = self._step(batch)
        return {"val_loss": loss.detach().cpu()}

    def validation_epoch_end(self, outputs):
        """End of validation epoch

        :param outputs: the output of the validation step
        """
        avg_loss = torch.stack([x["val_loss"].detach() for x in outputs]).mean()
        ### run the mcqa evaluation
        self.model_logger.info("validation_epoch_end!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if self.hparams.callback_monitor == "val_score":
            out_score = self.evaluate_output()
            class_score = torch.from_numpy(np.array(out_score)).detach().cpu()
        else:
            class_score = None

        ### 
        if class_score is not None:  tensorboard_logs = {"val_loss": avg_loss,"val_score" : class_score}
        else: tensorboard_logs = {"val_loss": avg_loss}        
        #return
        return _recursive_detach({"avg_val_loss": avg_loss,
                                      "log": tensorboard_logs,
                                      'progress_bar': tensorboard_logs})

    ## TRAINING STEPS AND EPOCHS
    
    def training_step(self,batch,batch_idx):
        """Runs a single training step

        :param batch: the target batch 
        :param batch_idx: the path id 
        :rtype: dict
        :returns: dictionary that includes loss 
        """
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss.detach().cpu()}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        """Called at the end of the training epoch 

        :param outputs: the outputs of the train step
        """
        avg_train_loss = torch.stack([x["loss"].detach() for x in outputs]).mean().detach().cpu().item()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        self._regenerate_data()
        return {"avg_train_loss": avg_train_loss,"log": tensorboard_logs,'progress_bar': tensorboard_logs}

    def _regenerate_data(self):
        pass

    ## TRAIN AND VALIDATION DATA LOADERS 

    def train_dataloader(self):
        """Loads the training dat for the step

        :returns: a data loader
        """
        ## data loader
        self.model_logger.info('Building train data loader, datadir=%s, batch_size=%d, drop_last=%s,shuffle=%s, num_epochs=%s' %\
                                   (self.hparams.data_dir,self.hparams.train_batch_size,self.hparams.drop_last,
                                        not self.hparams.no_shuffle,str(self.hparams.num_train_epochs)))

        dataloader = self.generic_dataloader(
            "train",
            final_eval=False,
            shuffle=not self.hparams.no_shuffle,
            batch_size=self.hparams.train_batch_size
        )

        ##move to configure_optimizers
        try:
            self.lr_scheduler
        except AttributeError:
            self.model_logger.info('Setting up the lr scheduler')
            t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs))
                    
            scheduler = get_linear_schedule_with_warmup(
                self.opt,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=t_total
            )
            
            self.lr_scheduler = scheduler

        return dataloader

    def val_dataloader(self):
        """Loader for the validation data 
        
        :returns: validation data loader 
        """
        return self.generic_dataloader("dev",final_eval=False,shuffle=False,
                                           batch_size=self.hparams.eval_batch_size)

    def generic_dataloader(self,name,final_eval=False,shuffle=False,batch_size=1):
        """Generic dataset loader. TODO: collapse all other loaders into this one 

        :returns: data loader 
        """
        self.model_logger.info('Loading generic file, name=%s, final_eval=%s, shuffle=%s, batch_size=%s' %\
                                   (name,final_eval,shuffle,batch_size))

        ## will need to update for tpu
        dataset = self._get_data(name,final_eval=final_eval)
        return DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        drop_last=False,
                        num_workers=self.hparams.num_workers)

    ## general method for collecting data
    def _get_data(self,split,final_eval=False):
        """Main method for getting the data 

        :param split: the particular split 
        :returns: dataset 
        """
        return self.dclass.from_file(self.hparams,
                                         self._tokenizer,
                                         split,
                                         final_eval)

    @property
    def tokenizer(self):
        """Returns the model tokenizer

        :returns: the underlying tokenizer
        """
        return self._tokenizer

    @property
    def model(self):
        """Returns the underlying T5 model

        :returns: the underlying T5 model 
        """
        return self._model

    ## tqdm
    
    def get_tqdm_dict(self):
        tqdm_dict = {
            "loss": "{:.3f}".format(self.trainer.avg_loss.detach().cpu()),
            "lr": self.lr_scheduler.get_last_lr()[-1].detach().cpu()}

    def is_logger(self):
        ## single gpu
        return self.trainer.global_rank <= 0

    ### special logger (ConfigurableClass conflicts with plt module here)

    @property
    def model_logger(self):
        """Returns a logger instance

        """
        level = '.'.join([__name__,type(self).__name__])
        return logging.getLogger(level)

    @classmethod
    def from_config(cls,config):
        """Load from a global configuration 

        :param config: the global configuration 
        :returns: Conditional T5 model 
        """
        ## build model and tokenizer
        return cls(config)

    ### generation

    def _classification_step(self,batch,
                                 max_length,
                                 no_repeat_ngram_size=None,
                                 num_beams=None,
                                 do_sample=None,
                                 top_p=None,
                                 min_length=None,
                                 top_k=None,
                                 num_return_sequences=None,
                                 ):
        """Run the classifier portion of T5 

        Decoder parameters are uses to make consistent with generation step, 
        but are not used here.

        :param max_length: set by each item 
        :param no_repeat_ngram_size: the maximum n-gram penalty (default is 2)
        :param num_beams: the number of beams to use 
        :param top_p: the nucleaus sampling parameter
        :param min_length: the minimum length of the output 
        :param top_k: the top k items to sample from 
        :rtype: dict 
        :returns: T5 model outputs
        """
        outs = self.model.generate(input_ids=batch["source_ids"].to(self._device),
                                    attention_mask=batch["source_mask"].to(self._device),
                                    max_length=max_length)
                                    #max_length=self.hparams.classification_length)
        return outs

    #_generative_step = _classification_step

    def _generative_step(self,batch,
                             max_length,
                             no_repeat_ngram_size=None,
                             num_beams=None,
                             do_sample=None,
                             top_p=None,
                             min_length=None,
                             top_k=None,
                             num_return_sequences=None):
        """Run T5 as a full generation model for a step using decoder 
        presets specified. 

        :param max_length: set by each item 
        :param no_repeat_ngram_size: the maximum n-gram penalty (default is 2)
        :param num_beams: the number of beams to use 
        :param top_p: the nucleaus sampling parameter
        :param min_length: the minimum length of the output 
        :param top_k: the top k items to sample from 
        :rtype: dict 
        """
        ## use defaults if not set 
        no_repeat_ngram_size = self.hparams.no_repeat_ngram_size if no_repeat_ngram_size is None else \
          no_repeat_ngram_size
        num_beams = num_beams if num_beams is not None else self.hparams.num_beams
        do_sample = do_sample if do_sample else self.hparams.do_sample
        top_p = top_p if top_p is not None else self.hparams.top_p
        top_k = top_k if top_k is not None else self.hparams.top_k
        if do_sample and top_p: top_k = 0
        elif do_sample and top_k: top_p = None

        outs = self.model.generate(input_ids=batch["source_ids"].to(self._device),
                                    attention_mask=batch["source_mask"].to(self._device),
                                    max_length=max_length,
                                    min_length=min_length,
                                    num_beams=num_beams,
                                    early_stopping=True, ## <---- look at this, eos_
                                    no_repeat_ngram_size=no_repeat_ngram_size,
                                    top_p=top_p,
                                    top_k=top_k,
                                    do_sample=do_sample,
                                    num_return_sequences=num_return_sequences,
                                    use_cache=True,
                                    return_dict_in_generate=True,
                                    output_scores=True
                                   )
        input_id_list = batch["source_ids"][0].tolist()
        tokens=self._tokenizer.convert_ids_to_tokens(input_id_list)
        out_dic = {k: outs[k] for k in outs.keys()}
        out_dic['tokens'] = tokens

        return out_dic

    @torch.no_grad()
    def evaluate_output(self,dtype='dev',final_eval=False):
        """Method for evaluating output, called after training step (passes by default). Loads the 
        data manually so it provides more control over printing model output.
        
        :param final_eval: is the final evaluation?
        :rtype final_eval: bool
        :rtype: pytorch.Tensor
        """
        raise NotImplementedError

    @classmethod
    def load_existing(cls,config):
        if wandb_available and config.wandb_model:
            _grab_wandb_data(config)

        if not config.target_model or not os.path.isdir(config.target_model):
            raise ValueError('Must specify model path')

        ## load hyper-parameters
        parser = ArgumentParser()
        args = parser.parse_args([])
        hparams_file = os.path.join(config.target_model,"commandline_args.txt")
        with open(hparams_file,'r') as f: args.__dict__ = json.load(f)

        ## update (needs work!)
        _update_config(args,config)
        util_logger.info("updated parameters: %s" % str(args))

        ## build model 
        model = cls(args,config.target_model,config.target_model)
        if config.special_device and torch.cuda.is_available():
            model.to(config.special_device)
        return model
    
    def query(self,text_input,prefix='answer:'):
        """Query the model via a text query 

        :param text_input: the text query 
        :param prefix: the text prefix for the model
        """
        raise NotImplementedError

    def __del__(self):
        model_keys = [k for k in self.__dict__]
        for key in model_keys:
            value = self.__dict__[key]
            try: value.to("cpu")
            except: pass 
            del self.__dict__[key]

class CustomTrainer(pl.Trainer):
    def __del__(self):
        model_keys = [k for k in self.__dict__]
        for key in model_keys:
            value = self.__dict__[key]
            try: value.to("cpu")
            except: pass 
            del self.__dict__[key]


################
# WANDB STUFF  #
################

def _grab_wandb_data(config):
    """Downloads the wandb data and use it as main data 

    :param config: the global configuration 
    """
    init_wandb(config)

    if not config.wandb_model and not config.wandb_data:
        return

    ## cache paths
    dfile_type = config.wandb_data.split("/")[-1]
    data_cache = os.path.join(WANDB_CACHE,dfile_type)
    mfile_type = config.wandb_model.split("/")[-1]
    model_cache = os.path.join(WANDB_CACHE,mfile_type)
    
    with wandb.init(entity=config.wandb_entity) as run:
        ## grab the data
        if config.wandb_data:
            artifact = run.use_artifact(config.wandb_data, type='dataset')
            artifact_dir = artifact.download(root=data_cache)
            util_logger.info('Download data to: %s,eval_subdir=%s' % (artifact_dir,config.eval_subdir))
            if config.eval_subdir and config.eval_subdir != "/":
                config.data_dir = os.path.join(artifact_dir,config.eval_subdir)
            else: 
                config.data_dir = artifact_dir

        ## grab existing model if specified
        if config.wandb_model:
            model = run.use_artifact(config.wandb_model, type='model')
            model_dir = model.download(root=model_cache)
            util_logger.info('Download data to: %s' % model_dir)
            config.target_model = model_dir

        ## turn off 
        config.wandb_model = ""

def init_wandb_logger(config):
    """Initializes the wandb logger 

    :param config: the global configuration 
    """
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_name
    )
    return wandb_logger

def _push_wandb_experiment(config,metrics):
    """Generic method for pushing the wandb data 

    :param config: the global experiment configuration 
    :param metrics: the resulting metrics 
    """
    wandb.log(metrics)
    wandb.log({"model_name" : config.model_name, "eval_name" : config.eval_name})
    run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name="",
    )

    ## back up the standard T5 model files if specified
    if config.save_wandb_model and not config.no_training:
        util_logger.info('Backing up the model files to wandb')
        ## save instead as an artifact
        martifact = wandb.Artifact('%s_model' % config.wandb_name, type='model')

        for model_file in [
                "added_tokens.json",
                "commandline_args.txt",
                "config.json",
                "pytorch_model.bin",
                "special_tokens_map.json",
                "spiece.model",
                "tokenizer_config.json",
                ]:
            martifact.add_file(
                os.path.join(config.output_dir,model_file)
            )
            
        ## log model 
        run.log_artifact(martifact)

    ## back up the output file if exists 
    if config.print_output:
        util_logger.info('Trying to log model output...')
        artifact = wandb.Artifact('%s_out' % config.wandb_name.replace(">","-"), type='model_output')
        artifact.add_file(os.path.join(config.output_dir,"dev_eval.tsv"))

        # Back up the attention files if exists.
        if config.attention_local_dir:
            util_logger.info(f'Trying to back up in wandb attention from dir: {config.attention_local_dir}')
            files = [f for f in os.listdir(config.attention_local_dir)]
            for file_path in files:
                full_path = os.path.join(config.attention_local_dir, file_path)
                util_logger.info(f'!!!!!!!!!!!!!!!:  {full_path},{file_path}')
                artifact.add_file(full_path)
        run.log_artifact(artifact)

    ### 
    run.finish()

def _remove_models(config):
    """Remove models specified

    :param config: the global experiment configuration 
    """
    for out_file in os.listdir(config.output_dir):
        if '.ckpt' in out_file and config.remove_checkpoints:
            logger.info('removing: %s' % out_file)
            os.remove(os.path.join(config.output_dir,out_file))
        elif 'pytorch_model' in out_file and config.remove_models:
            logger.info('removing: %s' % out_file)
            os.remove(os.path.join(config.output_dir,out_file))

#############
# TRAINERS  #
#############

class T5Trainer(ConfigurableClass):
    """Main trainer class"""
    _MODELS = {}

    def __init__(self,model,trainer):
        """Loads a trainer instance 

        :param model: the underlying T5 model 
        :param trainer: the trainer module 
        """
        self.logger.info('Loading the trainer')
        self._model = model
        self._trainer = trainer

    @property
    def model(self):
        return self._model

    @property
    def trainer(self):
        return self._trainer

    def train(self):
        """Main training loop 

        :returns: the best dev score  
        """
        self.logger.info('Started to train the model....')
        try: self._trainer.fit(self._model)
        ## the remaining issue on tpu https://github.com/PyTorchLightning/pytorch-lightning/issues/1637
        except FileNotFoundError as e: self.logger.error(e,exc_info=True)
        best_model_path = self._trainer.checkpoint_callback.best_model_path
        if not best_model_path: best_dev_score=-1.
        else: best_dev_score = self._trainer.checkpoint_callback.best_model_score.detach().item()
        self.logger.info('Best dev score: %s' % str(best_dev_score))
        return best_dev_score

    __call__ = train

    @classmethod
    def from_config(cls,config):
        """Load a trainer from a global configuration 

        :param config: the global config
        :raises: ValueError 
        """
        args = argparse.Namespace(**config.__dict__)
        mode = "max" if args.callback_monitor == "val_score" else "min"

        ## set up the seed
        set_seed(config.seed)

        ## set up the checkpoint
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=args.output_dir,
            prefix=args.callback_prefix,
            monitor=args.callback_monitor,
            mode=mode,
            save_top_k=args.save_top_k,
            verbose=args.verbose,
            period=args.period
        )

        ## early stopping callback (if available)
        early_stop_callback=False 
        if config.early_stopping:

            ## early stopping callback 
            early_stop_callback = pl.callbacks.EarlyStopping(
                monitor=args.callback_monitor,
                min_delta=0.00,
                patience=args.patience,
                verbose=args.verbose,
                mode=mode)

        ## load the training parameters
        reload_data = True if args.T5_type == "QuestionContextGenerator" else False
            
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=args.n_gpu,
            max_epochs=args.num_train_epochs,
            deterministic=args.deterministic,
            early_stop_callback=early_stop_callback,
            precision=16 if args.fp_16 else 32,
            amp_level=args.opt_level,
            gradient_clip_val=args.max_grad_norm,
            checkpoint_callback=checkpoint_callback,
            callbacks=[T5LoggingCallback()],
            auto_lr_find=args.auto_lr_find,
            reload_dataloaders_every_epoch=reload_data,
            num_sanity_val_steps=0,
            log_gpu_memory='all'
        )

        if args.tpu_cores > 0:
            train_params["tpu_cores"] = args.tpu_cores
        if config.wandb_project and wandb_available:
            train_params['logger'] = init_wandb_logger(config)

        ## set up the model
        mclass = cls._MODELS.get(args.T5_type,None)
        if mclass is None:
            raise ValueError(
                'Unknown T5 model types: %s' % args.T5_type
            )

        ## load an existing model
        if args.target_model:
            util_logger.info('Loading an existing model: %s' % args.target_model)
            model = mclass.load_existing(args)
        else: 
            model = mclass.from_config(args)

        ### trainer 
        trainer = CustomTrainer(**train_params)
        return cls(model,trainer)

    ## context manager stuff
    ## with large models memory needs to be handled properly here

    def force_exit(self):
        ## current a hack to deal with some strange memory issues  
        self.logger.info('Attempting to exit trainer, trying to clear CUDA cache, device=%s' % self._model._device)

        self._model._device = torch.device('cuda')
        try:
            target_device = self._model._device
            self._model.to("cpu")
            del self._trainer
            del self._model
            del self
            with torch.cuda.device(target_device): torch.cuda.empty_cache()
            ## manual garbage collection
            import gc
            gc.collect()

        except Exception as e:
            self.logger.error('Error with manual forced exit',exc_info=True)

    def __enter__(self):
        self.logger.info('Entering the trainer...')
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass



def params(config):
    """Main parameters for running the T5 model

    :param config: the global configuration object
    """
    from transformer_tools.model import params as mparams
    mparams(config)

    group = OptionGroup(config,"transformer_tools.T5Base",
                            "KnowledgeManager settings")

    
    group.add_option("--dtype",
                          dest="dtype",
                          default='mcqa',
                          help="The type of dataset to train [default='']")

    group.add_option("--opt_level",
                          dest="opt_level",
                          default='01',
                          type=str,
                          help="The optional level  [default='']")

    group.add_option("--evaluator",
                         dest="evaluator",
                         default='single_token',
                         help="The type of evaluator to use [default='single_token']")

    
    group.add_option("--full_answer",
                         dest="full_answer",
                         action='store_true',
                         default=False,
                         help="predict full answers [default=False]")

    group.add_option("--add_explanations",
                         dest="add_explanations",
                         action='store_true',
                         default=False,
                         help="Add explanations to the answers [default=False]")

    group.add_option("--no_special_tokens",
                         dest="no_special_tokens",
                         action='store_true',
                         default=False,
                         help="Dont add special tokens to model [default=False]")

    group.add_option("--new_tokens",
                         dest="new_tokens",
                         type=str,
                         default='',
                         help="Special tokens to add, delimited by `;` [default='']")

    group.add_option("--mark_knowledge",
                         dest="mark_kwnowledge",
                         action='store_true',
                         default=False,
                         help="Mark the type of knowledge being used [default=False]")

    group.add_option("--deterministic",
                         dest="deterministic",
                         action='store_true',
                         default=False,
                         help="Make training deterministic [default=False]")
    
    group.add_option("--max_seq_length",
                         dest="max_seq_length",
                         default=512,
                         type=int,
                         help="size of decoding  [default=2]")

    group.add_option("--callback_prefix",
                         dest="callback_prefix",
                         default="checkpoint",
                         type=str,
                         help="batch size [default='checkpoint']")

    group.add_option("--period",
                         dest="period",
                         default=1,
                         type=int,
                         help="the period (number of epochs) between checkpoints [default=1]")

    group.add_option("--save_top_k",
                         dest="save_top_k",
                         default=1,
                         type=int,
                         help="Number of models to save [default=1]")

    group.add_option("--drop_last",
                         dest="drop_last",
                         action='store_true',
                         default=False,
                         help="Drop the last batches [default=False]")

    group.add_option("--generate_once",
                         dest="generate_once",
                         action='store_true',
                         default=False,
                         help="Generate context only once [default=False]")

    group.add_option("--no_generate_train",
                         dest="no_generate_train",
                         action='store_true',
                         default=False,
                         help="Leave training data as-is [default=False]")

    group.add_option("--print_bleu",
                         dest="print_bleu",
                         action='store_true',
                         default=False,
                         help="Print the bleu score (if model does generation) [default=False]")

    group.add_option("--add_prefix",
                         dest="add_prefix",
                         action='store_true',
                         default=False,
                         help="Add special prefix to data [default=False]")

    group.add_option("--split_explanations",
                         dest="split_explanations",
                         action='store_true',
                         default=False,
                         help="Add explanations separately from classification labels [default=False]")

    group.add_option("--num_workers",
                         dest="num_workers",
                         default=4,
                         type=int,
                         help="number of number of processes when loading data [default=4]")

    group.add_option("--max_answer",
                         dest="max_answer",
                         default=2,
                         type=int,
                         help="the maxium answer size [default=2]")

    group.add_option("--classifier_length",
                         dest="classifier_length",
                         default=4, ##<-- new tokenization 
                         type=int,
                         help="The space needed for classification [default=2]")

    group.add_option("--max_explanation",
                         dest="max_explanation",
                         default=150,
                         type=int,
                         help="the maximum size of the explanation [default=150]")

    group.add_option("--num_facts",
                         dest="num_facts",
                         default=150,
                         type=int,
                         help="the maximum number of facts to use [default=8]")


    ## decoder settings
    group.add_option("--retrain_batch",
                         dest="retrain_batch",
                         default=16,
                         type=int,
                         help="The batch for retraining [default=16]")

    group.add_option("--num_beams",
                         dest="num_beams",
                         default=3,
                         type=int,
                         help="The number of beams to use during search/full generation [default=3]")

    group.add_option("--do_sample",
                         dest="do_sample",
                         action='store_true',
                         default=False,
                         help="Use sampling instead of beam search [default=False]")

    group.add_option("--no_repeat_ngram_size",
                         dest="no_repeat_ngram_size",
                         default=2,
                         type=int,
                         help="Do not repeat ngrams of size greater than this [default=2]")

    group.add_option("--early_stop_decoding",
                         dest="early_stop_decoding",
                         action='store_true',
                         default=True,
                         help="Early stopping during decoding [default=True]")

    group.add_option("--top_p",
                         dest="top_p",
                         default=None,
                         help="Nucleaus sampling parameter [default=None]")

    group.add_option("--top_k",
                         dest="top_k",
                         default=None,
                         help="Another sampling parameter [default=None]")

    group.add_option("--min_length",
                         dest="min_length",
                         default=None,
                         help="Minimal length for the generation [default=None]")

    group.add_option("--regenerate_eval",
                         dest="regenerate_eval",
                         action='store_true',
                         default=False,
                         help="Use generation at eval time [default=False]")


    group.add_option("--regen_k",
                         dest="regen_k",
                         default=3,
                         type=int,
                         help="The number of sentences to sample/generate for generative training [default=5]")

    group.add_option("--attention_local_dir",
                         dest="attention_local_dir",
                         default="attention",
                         help="The local directory of the attention")

    config.add_option_group(group)


### the model seed

def set_seed(seed):
    """Sets the random seed 
 
    :param seed: the initial seed for randomization 
    :type seed: int
    :rtype: None 
    """
    util_logger.info('Setting up the random seed, seed=%d' % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

logger = logging.getLogger(__name__)

def run_trainer_tester(config,trainer_class,t5_class,eval_map={}): 
    """Run the full trainer tester pipeline 
    
    :param config: the global configuration 
    """
    ## wandb stuff
    if wandb_available and config.wandb_project and wandb_available:
        init_wandb(config)
        ## download wandb data?
        if config.wandb_data: _grab_wandb_data(config)
    
    if not config.wdir and not config.output_dir:
        raise ValueError('Must specify a working directory using either `--wdir` or --outputdir')
    if not config.data_dir or not os.path.isdir(config.data_dir):
        raise ValueError('Must specify a valid data directory: %s' % config.data_dir)

    if not config.wdir and not config.output_dir:
        raise ValueError('Must specify a working directory using either `--wdir` or --outputdir')
    if not config.data_dir or not os.path.isdir(config.data_dir):
        raise ValueError('Must specify a valid data directory: %s' % config.data_dir)
    
    ### training (if set)
    best_dev_score = -1.
    metrics  = {}
    wandb_logger = None 
    if not config.no_training:

        with trainer_class.from_config(config) as trainer:
            best_dev_score = trainer()
            config.target_model   = config.output_dir
            config.special_device = trainer._model.device

            ## hack 
            trainer.force_exit()
            metrics[eval_map.get("best_dev_score","best_dev_score")] = best_dev_score

    elif wandb_available and config.wandb_project:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_name,
        )
        #wandb_logger = _init_wandb_logger(config)

    ## evaluation (if set) 
    if config.dev_eval or config.test_eval or config.train_eval:

        ###
        util_logger.info('Going into evaluation branch...')
        model    = t5_class.load_existing(config)
        model.eval()
        util_logger.info('loading model (might take time)...')

        ## fix pointers
        model.hparams.data_dir   = config.data_dir
        model.hparams.output_dir = config.output_dir
        print_output = config.print_output

        # if wandb_logger:
        #     wandb_logger.watch(model.model)

        ## (moved this out of the trainer) 
        if config.train_eval:
            util_logger.info('Evaluating train...')
            train_eval_score = model.evaluate_output(dtype='train',final_eval=print_output)
            metrics[eval_map.get("train_eval","train_eval")] = train_eval_score

        if config.dev_eval:
            util_logger.info('Evaluating dev...')

            ## regenerate evaluation data? 
            if config.regenerate_eval: model.regenerate_eval(split="dev")
            dev_eval_score = model.evaluate_output(dtype='dev',
                                                   final_eval=print_output,
                                                   attention_local_dir=config.attention_local_dir)
            metrics[eval_map.get("dev_eval","dev_eval")] = dev_eval_score
            if eval_map.get("best_dev_score","best_dev_score") not in metrics:
                metrics[eval_map.get("best_dev_score","best_dev_score")] = dev_eval_score

        if config.test_eval:
            util_logger.info('Evaluating test...')
            ## print test output
            if config.regenerate_eval: model.regenerate_eval(split="test")
            test_eval_score = model.evaluate_output(dtype='test',
                                                    final_eval=print_output,
                                                    attention_local_dir=config.attention_local_dir)
            metrics[eval_map.get("test_eval","test_eval")] = test_eval_score

    else:
        metrics[eval_map.get("dev_eval","dev_eval")] = best_dev_score

    ## print metrics
    util_logger.info('Attempting to write metrics file...')
    if config.output_dir and os.path.isdir(config.output_dir):

        metrics_out = os.path.join(config.output_dir,"metrics.json")
        util_logger.info('Attempting to write metrics file, out=%s...' % metrics_out)
        ## print out 
        with open(metrics_out,'w') as my_metrics:
            my_metrics.write(json.dumps(metrics,indent=4))

        ### wandb 
        if wandb_available and config.wandb_project:
            _push_wandb_experiment(config,metrics)

    ## remove models (if desired)
    if config.remove_models or config.remove_checkpoints:
        util_logger.info('Attempting to remove models...')
        _remove_models(config)

def main(argv):
    """The main execution point for running the T5 model 

    :param argv: the cli arguments 
    :raises: ValueError
    """
    from transformer_tools import initialize_config

    ## set up config and get working directory set up
    config = initialize_config(argv,params)

    ## run 
    run_trainer_tester(config)
