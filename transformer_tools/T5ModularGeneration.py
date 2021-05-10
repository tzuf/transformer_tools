import os
import pathlib
import tempfile
import time
import logging
import re
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import wandb
from dataclasses import dataclass
from transformer_tools import initialize_config
from optparse import OptionParser,OptionGroup
from transformer_tools.util.t5_util import *
from transformer_tools.util.aaac_util import aaac_eval
from transformer_tools.T5Generative import T5GenenerationModel,params

from transformer_tools.T5Base import (
    Text2TextData,
    T5LoggingCallback,
    T5Trainer,
    T5Text2TextBase,
    run_trainer_tester,
)



util_logger = logging.getLogger('transformer_tools.T5Generative')



class T5ModularGenenerationModel(T5GenenerationModel):
    """This class implements modular generation and multi-angular evaluation

    NOTE : made for AAAC

    """


    MODES = [
        {'id':'s => a','from':['argument_source'],'to':'argdown_reconstruction'},
        {'id':'s+r => a','from':['argument_source','reason_statements'],'to':'argdown_reconstruction'},
        {'id':'s+c => a','from':['argument_source','conclusion_statements'],'to':'argdown_reconstruction'},
        {'id':'r+c => a','from':['reason_statements','conclusion_statements'],'to':'argdown_reconstruction'},
        {'id':'s+r+c => a','from':['argument_source','reason_statements','conclusion_statements'],'to':'argdown_reconstruction'},
        {'id':'s => r','from':['argument_source'],'to':'reason_statements'},
        {'id':'s+a => r','from':['argument_source','argdown_reconstruction'],'to':'reason_statements'},
        {'id':'s+c => r','from':['argument_source','conclusion_statements'],'to':'reason_statements'},
        {'id':'s => c','from':['argument_source'],'to':'conclusion_statements'},
        {'id':'s+a => c','from':['argument_source','argdown_reconstruction'],'to':'conclusion_statements'},
        {'id':'s+r => c','from':['argument_source','reason_statements'],'to':'conclusion_statements'},
    ]

    
    @torch.no_grad()
    def evaluate_modular_output(self,
        data_raw:pd.DataFrame=None,
        mode_sequence=[],
        dtype='dev',
        final_eval=False
    ):
        """Method for modular generation and evaluating output. Loads the raw AAAC data manually.
        
        :param dataset: raw aaac dataset
        :rtype dataset: pandas.DataFrame
        :param mode_sequences: mode sequence to follow for modular generation
        :rtype final_eval: [str]
        :param final_eval: is the final evaluation?
        :rtype final_eval: bool
        :rtype: pytorch.Tensor
        """
        self.model_logger.info('Attempting to evaluate input with dtype=%s,mode_sequence=%s,final_eval=%s,device=%s' %\
                                   (dtype,mode_sequence,final_eval,self._device))

        current_stack = data_raw.copy() # copy of data which will be updated during modular generation

        for i,mode_id in enumerate(mode_sequence):
            current_mode = next(m for m in self.MODES if m['id']==mode_id)
            prefix = current_mode['to']

            # construct specific input dataset for this step in modular generation
            ## construct prompt
            def inquire_prompt(row):
                inquire_prompt = ""
                for from_key in current_mode['from']:
                    inquire_prompt = inquire_prompt + ("%s: %s " % (from_key,row[from_key]))
            ## list of input prompts (string)                
            input_prompts = current_stack.apply(inquire_prompt,axis=1).tolist()
            ## tokenize
            inputs,targets = multi_query(input_prompts,self.tokenizer,self.hparams,prefix)
            ## load a dataset and loader 
            dataset = Text2TextData(inputs,targets)
            loader = DataLoader(dataset,
                                    batch_size=self.hparams.eval_batch_size,
                                    shuffle=False,
                                    num_workers=self.hparams.num_workers)

            # generate output

            ## list to store generated output
            outputs = []
            ofile = None if (not final_eval or not dataset.data_rep or not self.hparams.output_dir) else \
            os.path.join(self.hparams.output_dir,"%s_eval.tsv" % dtype)

            ## run the
            ## check mode
            gen_func = self._classification_step if not final_eval else self._generative_step
            output_size = self.hparams.max_answer if final_eval else self.hparams.classifier_length

            ## go through the batches 
            self.model_logger.info('Going through batches, ofile=%s, lenght of data rep=%d, func=%s, output_size=%d' %\
                                    (ofile,len(dataset.data_rep),gen_func.__name__,output_size))                                   
            for batch in tqdm(loader):
                outs = gen_func(batch,max_length=output_size,
                                    no_repeat_ngram_size=self.hparams.no_repeat_ngram_size,
                                    num_beams=self.hparams.num_beams,
                                    do_sample=self.hparams.do_sample,
                                    top_p=self.hparams.top_p,
                                    min_length=self.hparams.min_length,
                                    top_k=self.hparams.top_k)

                dec    = [self.tokenizer.decode(ids.detach().cpu()) if self.tokenizer.decode(ids).strip() else "" for ids in outs]
                outputs.extend(dec)

            # update current_stack with generated output

            current_stack[prefix] = outputs            


        ### pass to custom evaluator to make sense of it
        df_score = aaac_eval(output=current_stack,target=data_raw,mode_sequence=mode_sequence)

        if ofile: pd.concat([current_stack,df_score],axis=1).to_csv(ofile,sep="\t")
        self.model_logger.info('Processed and scored %d outputs/targets' % len(outputs))
        ## 
        return df_score






def main(argv):
    """Main execution loop

    :param argv: the cli input 
    """
    config = initialize_config(argv,params)

    ### 
    util_logger.info('Loading model (might take time)...')
    model    = T5ModularGenenerationModel.load_existing(config)
    model.eval()
 
    ## fix pointers
    model.hparams.data_dir   = config.data_dir
    model.hparams.output_dir = config.output_dir
    print_output = config.print_output

    ### load test data from wandb
    with tempfile.TemporaryDirectory() as tempdir: 
        api = wandb.Api()
        artifact = api.artifact(
            config.wandb_data, 
            type='raw_data'
        )
        artifact.download(root=tempdir)
        DATA_JSON=os.path.join(tempdir,"aaac.jsonl")
        df_aaac_test = pd.read_json(DATA_JSON,lines=True,orient='records')
    
    mode_sequence = ['s => a']

    df_modular_eval_scores = model.evaluate_modular_output(
        data_raw=df_aaac_test,
        mode_sequence=mode_sequence,
        dtype='dev',
        final_eval=print_output
    )

    print(df_modular_eval_scores.info())
    print(df_modular_eval_scores.head())

