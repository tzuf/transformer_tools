import os
import time
import datetime
import logging
import re
import sys
import json
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from dataclasses import dataclass
from transformer_tools import initialize_config
from optparse import OptionParser,OptionGroup
from transformer_tools.util.t5_util import *
from transformer_tools.util.t5_util import single_token_eval_with_proof
from torch.nn import CrossEntropyLoss

from transformer_tools.T5Base import (
    Text2TextData,
    T5LoggingCallback,
    T5Trainer,
    T5Text2TextBase,
    run_trainer_tester,
)

util_logger = logging.getLogger('transformer_tools.T5Classification')

_CLASSIFICATION_BUILDERS={
    "json_mcqa"       : json_mcqa,
    "multi_qa"        : multi_qa,
}

## data manager and builder 

class ClassificationText2Text(Text2TextData):
    """Text2Text data class specialized for classification

    :param DATA_BUILDERS: the type of data builders available for the 
      the classifier.
    """
    DATA_BUILDERS = _CLASSIFICATION_BUILDERS

## lightning module 

class T5Classification(T5Text2TextBase):
    """Class for Pretrained ConditionalGeneration T5 Models 
    that specifically supports classification. 

    :param EVALUATOR: the classifier evaluation code 
    :param LOADER: the classifier data loader class 
    """
    EVALUATOR = single_token_eval
    LOADER    = ClassificationText2Text

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
        """Replaces generative step with the classification step (no actual generation
        is used in this model) 

        :param max_length: set by each item 
        :param no_repeat_ngram_size: the maximum n-gram penalty (default is 2)
        :param num_beams: the number of beams to use 
        :param top_p: the nucleaus sampling parameter
        :param min_length: the minimum length of the output 
        :param top_k: the top k items to sample from 
        :rtype: dict 
        """
        return self._classification_step(batch,max_length)

class T5ClassificationExplanation(T5Classification):
    """This class implements explanations alongside a simple classifier

    NOTE : this is a specialized class tailored to my current Jsonl/GlossKB 
    file format. 

    :param EVALUATOR: the classifier evaluation code 
    :param LOADER: the classifier data loader class 
    """

    def prepare_data(self):
        """This method checks that provided data is an the right format
        for extracting explanations; it is a little hacky at this moment, and
        relies on a utility function. 
        
        :rtype: None 
        """
        self.model_logger.info('Preparing the explanation dataset...')
        special_tokens = prepare_explanations(self.hparams,self.tokenizer)
        ## special tokens
        if not self.hparams.no_special_tokens: self._add_special_tokens(special_tokens)
        ## backup
        self._backup_tokenizer()

    def _add_special_tokens(self,special_tokens=[]):
        """Adds special tokens based on provided list 

        :param token_list: the token list to cover
        """
        if not self.hparams.target_model and not self.hparams.no_special_tokens: 
            self.tokenizer.add_tokens(["[EXPL]"]+special_tokens)
            special_tokens_dict = {'additional_special_tokens': ['[EXPL]']+special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _generative_step(self,batch,
                             max_length,
                             no_repeat_ngram_size=None,
                             num_beams=None,
                             do_sample=None,
                             top_p=None,
                             min_length=None,
                             top_k=None,
                             num_return_sequences=None,
                             ):
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
        no_repeat_ngram_size = self.hparams.no_repeat_ngram_size if no_repeat_ngram_size is None else \
          no_repeat_ngram_size
        num_beams = num_beams if num_beams is not None else self.hparams.num_beams
        do_sample = do_sample if do_sample is not None else self.hparams.do_sample
        to_p = top_p if top_p is not None else self.hparams.top_p
        top_k = top_k if top_k is not None else self.hparams.top_k
        if do_sample and top_p: top_k = 0

        ### 
        outs = self.model.generate(input_ids=batch["source_ids"].to(self._device),
                                    attention_mask=batch["source_mask"].to(self._device),
                                    max_length=max_length,
                                    min_length=min_length,
                                    num_beams=num_beams,
                                    early_stopping=True,
                                    no_repeat_ngram_size=no_repeat_ngram_size,
                                    top_p=top_p,
                                    top_k=top_k,
                                    do_sample=do_sample,
                                    num_return_sequences=num_return_sequences)

        return outs


    @torch.no_grad()
    def evaluate_output(self,dtype='dev',final_eval=False,force_prefix=None):
        """Method for evaluating output, called after training step (passes by default). Loads the 
        data manually so it provides more control over printing model output.
        
        :param final_eval: is the final evaluation?
        :rtype final_eval: bool
        :rtype: pytorch.Tensor
        """
        self.model_logger.info('Attempting to evaluate input with dtype=%s, final_eval=%s,device=%s, max_answer=%d, no_repeat_ngram_size=%d, num_beams=%d, do_sample=%s, top_p=%s,force_prefix=%s' %\
                                   (dtype,final_eval,self._device,
                                   self.hparams.max_answer,
                                   self.hparams.no_repeat_ngram_size,
                                   self.hparams.num_beams,
                                   str(self.hparams.do_sample),
                                   self.hparams.top_p,
                                   force_prefix,
                                   ))

        ## loads the dataset on each round, can't seem to find the trainer dataset anywhere!
        dataset = self._get_data(dtype,final_eval=final_eval,force_prefix=force_prefix)
        loader = DataLoader(dataset,
                                batch_size=self.hparams.eval_batch_size,
                                shuffle=False,
                                num_workers=self.hparams.num_workers)

        outputs = []; targets = []
        ofile = None if (not final_eval or not dataset.data_rep or not self.hparams.output_dir) else \
          os.path.join(self.hparams.output_dir,"%s_eval.tsv" % dtype)
        if ".tsv" in ofile and self.hparams.print_json:
            ofile =  os.path.join(self.hparams.output_dir,"%s_eval.jsonl" % dtype)
          
        ## run the
        ## check mode
        gen_func = self._classification_step if not final_eval else self._generative_step
        output_size = self.hparams.max_answer if final_eval else self.hparams.classifier_length

        self.model_logger.info('Going through batches, ofile=%s, lenght of data rep=%d, func=%s, output_size=%d' %\
                                   (ofile,len(dataset.data_rep),gen_func.__name__,output_size))

        ## go through the batches 
        for batch in tqdm(loader):
            outs = gen_func(batch,max_length=output_size).detach().cpu()
            dec    = [self.tokenizer.decode(ids.detach().cpu()) if self.tokenizer.decode(ids).strip() else "" for ids in outs]
            target = [self.tokenizer.decode(ids.detach().cpu()) for ids in batch["target_ids"].detach()]
            outputs.extend(dec)
            targets.extend(target)

        ### pass to custom evaluator to make sense of it
        
        output_score, proof_score = self.evaluator(outputs,targets)
        if ofile: print_full_output(outputs,targets,dataset.data_rep,ofile,print_bleu=self.hparams.print_bleu)         
        self.model_logger.info('Processed and scored %d outputs/targets, score=%f, proof score=%f' % (len(outputs),output_score, proof_score))
        ## 
        return output_score, proof_score
    
class T5ClassificationMultiQA(T5ClassificationExplanation):
    """Allows data that is explicitly mixed with explanations, reasoning patterns, 
    QA pairs, ...

    :param EVALUATOR: the classifier evaluation code 
    :param LOADER: the classifier data loader class 
    """
    EVALUATOR = single_token_eval_with_proof

    def prepare_data(self):
        """This method checks that provided data is an the right format
        for extracting explanations; it is a little hacky at this moment, and
        relies on a utility function. 
        
        :rtype: None 
        """
        self.model_logger.info('Preparing the explanation dataset...')
        special_tokens = prepare_multi(self.hparams,self.tokenizer)
        self._add_special_tokens(special_tokens)
        self._backup_tokenizer()

    def query(self,text_input,prefix='answer:',
                  no_repeat_ngram_size=None,
                  num_beams=None,
                  do_sample=None,
                  top_p=None,
                  min_length=None,
                  num_return_sequences=None,
                  ):
        """Query the model via a text query 

        :param text_input: the text query 
        :param prefix: the text prefix for the model
        :param no_repeat_ngram_size: the maximum n-gram penalty (default is 2)
        :param num_beams: the number of beams to use 
        :param top_p: the nucleaus sampling parameter
        :param min_length: the minimum length of the output 
        :param top_k: the top k items to sample from 
        """
        inputs,targets = multi_query(text_input,self.tokenizer,self.hparams,prefix)

        ## load a dataset and loader 
        dataset = Text2TextData(inputs,targets)
        loader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=self.hparams.num_workers)

        gen_func = self._classification_step if prefix == "answer:" else self._generative_step
        output_size = self.hparams.classifier_length if prefix == "answer:" else self.hparams.max_answer
        self.model_logger.info('prefix=%s, output size=%d,gen func=%s' % (prefix,output_size,gen_func.__name__))

        for batch in loader:

            ### 
            outs = gen_func(batch,max_length=output_size,
                                no_repeat_ngram_size=no_repeat_ngram_size,
                                num_beams=num_beams,
                                do_sample=do_sample,
                                top_p=top_p,
                                min_length=min_length,
                                num_return_sequences=num_return_sequences)

            ## decoder output
            dec = [self.tokenizer.decode(ids).replace("<pad> ","").replace("</s>","").strip() \
                       if self.tokenizer.decode(ids).strip() else "" for ids in outs]

        return dec


class T5ClassificationMultiList(T5ClassificationMultiQA):
    EVALUATOR = single_token_list

class T5GenerativeTrainer(T5ClassificationExplanation):
    """Model runs generation during training 

    :param EVALUATOR: the classifier evaluation code 
    :param LOADER: the classifier data loader class 
    """
    EVALUATOR = single_token_eval

    def _regenerate_data(self):
        """Uses the model for doing generation, then collects the generated output and prepends to 
        model. 

        :returns: None 
        """
        #self.trainer.train_dataloader
        pass 
        exit('exiting...')
        
    def _rebuild_data(self,dtype='train'):
        """Regenerate data using generation 

        :param dtype: the type of data to rebuild 
        :returns: the dataset 
        """
        raise NotImplementedError
    

class QuestionContextGenerator(T5GenerativeTrainer):
    """T5 model that uses generation at training time to update 
    training and dev data. 

    notes: assumes a UnifiedQA style model to begin with. More specifically, 
    it should be a moel that has a mode `retrieve:` and `answer:` built-in

    :param EVALUATOR: the classifier evaluation code 
    :param LOADER: the classifier data loader class 
    """

    
    ## generate data before training
    
    def prepare_data(self):
        """Starts the generation process. 
        
        :rtype: None 
        """
        ## create link to original data
        self.hparams.orig_data = self.hparams.data_dir
        self.hparams.generation_step = 0
        self.instance_scores = {}

        ## make a new data directory 
        next_dir = os.path.join(self.hparams.output_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(next_dir)
        self.hparams.next_dir = next_dir

        ## first training data
        #self.regenerate_data("train")
        if not self.hparams.no_generate_train:
            self.regenerate_data("train")
        else:
            shutil.copy(os.path.join(self.hparams.data_dir,"train.jsonl"),next_dir)
            self.model_logger.info('Moved training to new directory untouched...')

        self.regenerate_data("dev")

        ## update data directory 
        self.hparams.data_dir = next_dir

        ## first backup tokenizer and hyper-parameters
        self.tokenizer.save_pretrained(self.hparams.output_dir)
        with open(os.path.join(self.hparams.output_dir,'commandline_args.txt'), 'w') as f:
            json.dump(dict(self.hparams.items()), f, indent=2)
            
    ## generate new data after epoch

    def _regenerate_data(self):
        """Uses the model for doing generation, then collects the generated output and prepends to 
        model. 

        :returns: None 
        """
        ## no more generation 
        if self.hparams.generate_once: return

        #self.trainer.train_dataloader
        next_dir = os.path.join(self.hparams.output_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(next_dir)
        self.hparams.next_dir = next_dir

        ## first training data
        if not self.hparams.no_generate_train:
            self.regenerate_data("train")
        else:
            shutil.copy(os.path.join(self.hparams.data_dir,"train.jsonl"),next_dir)
            self.model_logger.info('Moved training to new directory untouched...')
            
        self.regenerate_data("dev")
        self.hparams.data_dir = next_dir

        ## update dataset links
        self.model_logger.info('Updated the dataloaders...')

    def regenerate_eval(self,split='dev'):
        """Regenerate just for evaluation 
    
        :param split: the target split to regenerate
        """
        next_dir = os.path.join(self.hparams.output_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(next_dir)
        self.instance_scores = {}
        self.hparams.next_dir = next_dir
        self.regenerate_data("dev")
        self.hparams.data_dir = next_dir
        ## update dataset links
        self.model_logger.info('Updated the dataloaders...')

    @torch.no_grad()
    def question_scores(self,
        batch,
        decoder_input_ids=None,
        decoder_past_key_value_states=None,
        use_cache=True,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
    ):
        """Compute scores for a question batch. Quite similar to the models' 
        `forward` method, but cuts out some of the additional processing.
        
        """
        lm_labels = batch["target_ids"].to(self._device)
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        attention_mask = batch["source_mask"].to(self._device)
        
        input_ids = batch["source_ids"].to(self._device)
        decoder_attention_mask = batch['target_mask'].to(self._device)

        # Convert encoder inputs in embeddings if needed
        encoder_outputs = self._model.encoder(input_ids=input_ids,
                                                  attention_mask=attention_mask,
                                                  inputs_embeds=inputs_embeds,
                                                  head_mask=head_mask)

        hidden_states = encoder_outputs[0]

        if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._model._shift_right(lm_labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if decoder_past_key_value_states is not None:
            assert lm_labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self._model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_value_states=decoder_past_key_value_states,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
        )

        if use_cache is True:
            past = ((encoder_outputs, decoder_outputs[1]),)
            decoder_outputs = decoder_outputs[:1] + past + decoder_outputs[2:]

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self._model.model_dim ** -0.5)
        lm_logits = self._model.lm_head(sequence_output)

        if lm_labels is not None:
            ## `reduction` is important here, to not sum or take max
            loss_fct = CrossEntropyLoss(ignore_index=-100,reduction='none')
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))

        ## is this right? Seems to add zeros in odd spots.
        #return loss[loss > 0.]
        return loss

    @torch.no_grad()
    def regenerate_data(self,split):
        """Method for evaluating output, called after training step (passes by default). Loads the 
        data manually so it provides more control over printing model output.
        
        :rtype: None 
        """        ### training first 
        self.model_logger.info('Regenerating train set in %s, device=%s, max_answer=%d, no_repeat_ngram_size=%d, num_beams=%d, do_sample=%s, top_p=%s,top_k=%s,eval_batch_size=%d' %\
                                   (self.hparams.data_dir,self._device,
                                   self.hparams.max_answer,
                                   self.hparams.no_repeat_ngram_size,
                                   self.hparams.num_beams,
                                   self.hparams.do_sample,
                                   self.hparams.top_p,
                                   self.hparams.top_k,
                                   self.hparams.eval_batch_size,
                                   ))

        ## RUNNING THE GENERATION  
        ###########################
        
        ## debug settings
        inputs,targets,selected,full_data = retrain_loader(self.hparams,self.tokenizer,split)
        dataset = ClassificationText2Text(inputs,targets)
        batch_size = self.hparams.retrain_batch

        ## loader 
        loader = DataLoader(dataset,batch_size=batch_size,
                                shuffle=False,
                                num_workers=self.hparams.num_workers)
        output_size = self.hparams.max_answer

        ### run generator
        text_output = []
        num_return_sequences = self.hparams.regen_k if split == "train" else 1
        

        ### run the generatedx
        for batch in tqdm(loader):
            ## run the decoder
            outs = self._generative_step(batch,max_length=output_size,num_return_sequences=num_return_sequences)
            # ## decode the output 
            dec = [self.tokenizer.decode(ids.detach().cpu()) if self.tokenizer.decode(ids).strip()  else "" for ids in outs]
            ## slice into batches
            text_output += [dec[i:i+num_return_sequences] for i in range(0,len(dec),num_return_sequences)]

        ##RUNNING THE QA MOODE
        #########################

        self.model_logger.info('Running new contextual data through the QA model..')
        #### run classification mode with new generated data
        next_inputs,next_targets,next_selected = add_generated_text(self.hparams,self.tokenizer,selected,text_output)
        next_dataset = ClassificationText2Text(next_inputs,next_targets)
        next_loader = DataLoader(next_dataset,batch_size=batch_size,shuffle=False,num_workers=self.hparams.num_workers)

        ### 
        answer_out = []
        scores     = []
        outputs    = []
        targets    = []
        output_size = self.hparams.classifier_length
        
        ###
        for b,batch in enumerate(tqdm(next_loader)):
            target = [self.tokenizer.decode(ids.detach().cpu()) for ids in batch["target_ids"].detach()]

            ## only score if training 
            if split == "train":
                outs = self._classification_step(batch,max_length=self.hparams.classifier_length)
                dec = [self.tokenizer.decode(ids.detach().cpu()) if self.tokenizer.decode(ids).strip()  else "" for ids in outs]
                answer_out.extend(dec)
                targets.extend(target)
                
                full_outputs = self.question_scores(batch)
                logprobs = [s.detach().cpu().item() for i,s in enumerate(full_outputs) if i % 2 == 0]

                ### get -log prob score
                if len(logprobs) != len(dec):
                    self.model_logger.warning('Mismatched scores at batch %d, scores=%d, answers=%d' % (b,len(logprobs),len(dec)))
                scores += logprobs

            #### 
            else:
                answer_out.extend(target)
                targets.extend(target)
                scores += [0.0]*len(target)

        ### create final new dataset
        self.model_logger.info('Identifiers=%d,answers=%d,targets=%d,scores=%d' %\
                                   (len(next_selected),len(answer_out),len(targets),len(scores)))

        ## update 
        create_new_set(self.hparams,
                           next_selected,
                           answer_out,
                           targets,
                           scores,
                           full_data,split,
                           instance_scores=self.instance_scores
      )
        
    
_CLASSIFICATION_MODELS={
    "T5Classification"            : T5Classification,
    "T5ClassificationExplanation" : T5ClassificationExplanation,
    "T5ClassificationMultiQA"     : T5ClassificationMultiQA,
    "QuestionContextGenerator"    : QuestionContextGenerator,
    "T5ClassificationMultiList"   : T5ClassificationMultiList,
}
    

def T5Model(config): 
    """Factory method for loading a T5 model 

    :param config: the configuration 
    :returns: T5 model instance (loaded from configuration) 
    """
    mtype = _CLASSIFICATION_MODELS.get(config.T5_type,None)
    if mtype is None:
        raise ValueError('Unknown T5 Model...%s' % config.T5_type)
    return mtype
    

class T5ClassificationTrainer(T5Trainer):
    """Special trainer for T5 classification models


    :param _MODELS: Possible classifier models visible to the Classifier 
        trainer. 
    """
    _MODELS = _CLASSIFICATION_MODELS
    
def T5ClassificationModel(config):
    """Factory method for loading a T5 model 

    :param config: the configuration 
    :returns: T5 model instance (loaded from configuration) 
    """
    mtype = _CLASSIFICATION_MODELS.get(config.T5_type,None)
    if mtype is None:
        raise ValueError('Unknown T5 Model...%s' % config.T5_type)
    return mtype

def LoadModel(config):
    model_class = T5ClassificationModel(config)
    return model_class.load_existing(config)

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
                         default="val_score",
                         type=str,
                         help="batch size [default='val_score']")

    group.add_option("--T5_type",
                         dest="T5_type",
                         default='T5Classification',
                         type=str,
                         help="The type of T5 model to use [default=T5Classification]")

    group.add_option("--data_builder",
                         dest="data_builder",
                         default='multi_qa',
                         type=str,
                         help="Dataset builder function [default='json_mcqa']")
    
    group.add_option("--max_regenerate",
                         dest="max_regenerate",
                         default=500,
                         type=int,
                         help="The number of items to regenerate after each iteration (for generative training) [default=500]")

    group.add_option("--generation_prefix",
                         dest="generation_prefix",
                         default="retrieve:",
                         type=str,
                         help="The type of generation prefix to use [default='retrieve:']")

    config.add_option_group(group)

def main(argv):
    """Main execution point 

    :param argv: the main cli arguments
    :rtype: None 
    """
    ## config 
    config = initialize_config(argv,params)
    t5_class = T5Model(config)

    ## trainer testing
    run_trainer_tester(config, T5ClassificationTrainer, t5_class)
