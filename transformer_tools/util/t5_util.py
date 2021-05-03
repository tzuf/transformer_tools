### utility for T5
import re
import os
import sys
import json
import copy
import logging
import numpy as np
import torch
import string
import random
from tqdm import tqdm
from sklearn import metrics as sklearn_metrics
from nltk.translate.bleu_score import corpus_bleu
from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction

### sequence eval
import seqeval
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

__all__ = [
    "json_mcqa",
    "single_token_eval",
    "print_full_output",
    "prepare_explanations",
    "multi_qa",
    "prepare_multi",
    "multi_query",
    "generative_data",
    "bleu_eval",
    "print_full_with_bleu",
    "retrain_loader",
    "add_generated_text",
    "create_new_set",
    "full_seq_eval",
]

util_logger = logging.getLogger('transformer_tools.util.t5_util')

def _build_assertion_list(generated_context):
    assertions = []
    rel_type  = ''
    assertion = ''
    generate_context = re.sub(r'\s+|\n+',' ',generated_context)
    generated_context = "%s [EOS]" % generated_context
    
    for word in generated_context.split(" "):
        word = word.strip()
        rel_pattern = re.search(r'^([A-Z\_\-]+)$',word)
        subj_pattern = re.search(r'(.+)\:$',word)
        if rel_pattern:
            if rel_type:
                ## previous relation 
                assertion = assertion.strip()
                subj = ' '.join([w.strip() for w in assertion.split(":")[0].split()])
                if subj == assertion:
                    try: 
                        subj = assertion.split()[0]
                    except:
                        subj="unk"
                    subj = subj.translate(str.maketrans('', '', string.punctuation))
                    sentence = assertion
                else:
                    sentence = ' '.join([w for w in assertion.split(":")[1].split()])
                    assertions.append((rel_type,subj,sentence))
            rel_type = rel_pattern.groups()[0]
            if rel_type == "EOS": break
            assertion = ''
        else:
            assertion = "%s %s" % (assertion,word)
    return assertions


def create_new_set(config,identifiers,
                       answers,
                       targets,
                       scores,
                       original,
                       split,
                       instance_scores={},
                       ):
    new_items = {}
    assert len(identifiers) == len(answers) == len(targets) #== len(scores), "issue"

    ### add the best scoring ones
    for k,item in enumerate(identifiers):
        identifier = item["id"]

        ## add to overall score 
        if identifier not in instance_scores:
            instance_scores[identifier] = np.inf
        
        predicted_answer = answers[k]
        gold_answer = targets[k]
        if predicted_answer != gold_answer: continue
        answer_score = scores[k]
        ### added filter 
        
        if identifier not in new_items and answer_score <= instance_scores[identifier]:
            new_items[identifier]  = (item,answer_score)
            instance_scores[identifier] = answer_score

        elif identifier in new_items and answer_score <= new_items[identifier][-1] and\
          answer_score < instance_scores[identifier]:
            new_items[identifier]  = (item,answer_score)
            instance_scores[identifier] = answer_score

    util_logger.info('# of items with new context=%d' % len(new_items))
    new_out = os.path.join(config.next_dir,split+".jsonl")
    changed = 0

    with open(new_out,'w') as ofile: 
        for json_item in original:
            identifier = json_item["id"]
            if identifier in new_items:
                json_item = new_items[identifier][0]
                changed += 1
            else:
                if "$context$" not in json_item["question"]["stem"]:
                    json_item["question"]["stem"] = "$context$ empty %s" % json_item["question"]["stem"]
                    json_item["lf"] = "empty"

            json_item["prefix"] = "answer:"
            ofile.write(json.dumps(json_item))
            ofile.write("\n")
    util_logger.info('Printing to new output file: %s, # changed items=%d' % (new_out,changed))

def add_generated_text(config,tokenizer,selected,text_output):
    """Add items 

    """
    assert len(selected) == len(text_output), "wrong length"

    util_logger.info('Building QA set for %d items' % len(selected))
    identifiers = []
    inputs = []
    targets = []

    for k,json_line in enumerate(selected):
        ###
        for n,context in enumerate(text_output[k]):
            ###
            orig_context = context
            new_json = copy.deepcopy(json_line)
            #new_json["id"] = "%s %d" % (new_json["id"],n)
            patterns = ["%s %s" % (p[0],p[1]) for p in re.findall(r'([A-Z\_\-]+) ([a-z\s\\\/]+)\:',context)]
            if patterns:
                context = re.sub(r'[A-Z\_]{3,}','',re.sub(r'([A-Z\_\-]+) ([a-z\s\\\/]+)\:','. ',context))
            else:
                context = re.sub(r'([A-Z\_\-]+) ([a-z\s\\\/]+)\:',' ',context)
                
            ### remove answer generation (if it exists) 
            context = context.split("[EXPL] ")[-1].strip()

            ### get the actual patterns
            #if patterns:
            #new_json["full_context"] = orig_context
                # try: 
                #     #patterns = _build_assertion_list(context)
                # except Exception as e:
                #     util_logger.error("Encountered error parsing context...",exc_info=True)
            new_json["full_context"] = orig_context                
            new_json["lf"] = patterns

            ## remove previous contexts 
            if "$context$" in new_json["question"]["stem"]:
                new_json["question"]["stem"] = "$question$ %s" % new_json["question"]["stem"].split("$question$")[-1]

            new_json["question"]["stem"] = "$context$ %s %s" % (context,new_json["question"]["stem"])
            new_json["question"]["stem"] = re.sub(r'\s+',' ',re.sub(r'\. \. ','. ',new_json["question"]["stem"]))
            
            #new_json["question"]["stem"] = re.sub(r'\s+',' ',new_json["question"]["stem"])
            new_json["question"]["stem"] = new_json["question"]["stem"].replace("$context$ .","$context$ ")
            new_json["prefix"] = "answer:"
            identifiers.append(new_json)

            #### 
            input_ = "answer: %s </s>" % new_json["question"]["stem"]
            target = "%s </s>" % json_line["output"]

            for minput,moutput in [(input_,target)]:
                tokenized_inputs = tokenizer.batch_encode_plus(
                    [minput],
                    max_length=config.max_seq_length,
                    pad_to_max_length=True,
                    return_tensors="pt",
                    truncation=True, ## throws off warning if switched off
                )

                # tokenize targets
                tokenized_targets = tokenizer.batch_encode_plus(
                    [moutput],
                    #max_length=config.max_answer,
                    max_length=config.classifier_length,
                    pad_to_max_length=True,
                    return_tensors="pt",
                    truncation=True, ## throws off warning if switched off
                )

                inputs.append(tokenized_inputs)
                targets.append(tokenized_targets)

            if k <= 5:
                util_logger.info('************************')
                util_logger.info('INPUT: %s' % input_)
                util_logger.info('TARGET: %s' % target)

    ###
    return (inputs,targets,identifiers)
            
            

def retrain_loader(config,tokenizer,split):
    """Loading a dataset for the generative re-builder 
    
    """
    inputs = []
    targets = []
    data_rep = []
    selected = []
    full_data = []
    max_regenerate = config.max_regenerate
    util_logger.info('Running retrain re-loader. Max regenerate amount: %d' % max_regenerate)

    path = os.path.join(config.data_dir,split+".jsonl")
    util_logger.info('Reading data here: %s' % path)

    with open(path,"r",encoding="utf-8") as f:
        lines = [json.loads(l.strip()) for l in f.readlines()]
        random.shuffle(lines)

        ### try to put the examples without context in the front 
        no_context = [l for l in lines if "$context$ empty" in l["question"]["stem"]]
        context = [l for l in lines if "$context$ empty" not in l["question"]["stem"]]
        if (len(no_context) + len(context)) != len(lines):
            util_logger.info('Error with distributing context/no context')
        lines = no_context+context

        #####         

        for k,line in enumerate(lines):
            #json_line = json.loads(line.strip())
            json_line = line
            full_data.append(json_line)
            if k >= max_regenerate: continue

            #######
            stem = json_line["question"]["stem"]

            if "$context$" in stem:
                stem = "$question$ %s" % stem.split("$question$")[-1]

            identifier = "%s_%d" % (split,k) if "id" not in json_line else json_line["id"]
            #input_  = "retrieve: %s </s>" %  json_line["question"]["stem"].replace("$question$","").strip()
            input_  = "retrieve: %s </s>" %  stem.replace("$question$","").strip()
            input_ = re.sub(r'\s+|\n+',' ',input_)
            target = "%s </s>" % json_line["output"]
            selected.append(json_line)
            
            for minput,moutput in [(input_,target)]:
                ## featurize
                tokenized_inputs = tokenizer.batch_encode_plus(
                    [minput],
                    max_length=config.max_seq_length,
                    pad_to_max_length=True,
                    return_tensors="pt",
                    truncation=True, ## throws off warning if switched off
                )

                # tokenize targets
                tokenized_targets = tokenizer.batch_encode_plus(
                    [moutput],
                    max_length=config.max_answer,
                    #max_length=config.classifier_length,
                    pad_to_max_length=True,
                    return_tensors="pt",
                    truncation=True, ## throws off warning if switched off
                )

                inputs.append(tokenized_inputs)
                targets.append(tokenized_targets)

            if k <= 5: # and (split == "train" or final_eval): # and dtype == 'train':
                util_logger.info('************************')
                util_logger.info('INPUT: %s' % input_)
                util_logger.info('TARGET: %s' % target)


    ## shuffle 
    random.shuffle(full_data)
    return (inputs,targets,selected,full_data)

def generative_data(config,
                      tokenizer,
                      split,
                      final_eval):
    """Loading a dataset for the generative trainer 

    """
    main_prompt = ""
    split_file = os.path.join(config.data_dir,"%s.jsonl" % split)
    util_logger.info('Reading file: %s, final eval=%s' % (split_file,final_eval))
    if not os.path.isfile(split_file): raise ValueError('Unknown file: %s' % split_file)

    inputs       = []
    targets      = []
    data_rep     = []
    input_sizes  = []
    output_sizes = []
    original_sizes = [] 

    with open(split_file,"r",encoding="utf-8") as f:
        for k,line in enumerate(f):
            json_line = json.loads(line.strip())
            identifier = "%s_%d" % (split,k) if "id" not in json_line else json_line["id"]
            input_  = json_line["question"]["stem"]
            target = "%s </s>" % json_line["output"]
            original_sizes.append(len(target.split()))

            if "prefix" in json_line:
                input_ = "%s %s" % (json_line["prefix"],input_)

            for minput,moutput in [(input_,target)]:
                input_tokens = tokenizer.tokenize(minput)
                output_tokens = tokenizer.tokenize(moutput)
                input_sizes.append(len(input_tokens))
                output_sizes.append(len(output_tokens))

                ## featurize
                tokenized_inputs = tokenizer.batch_encode_plus(
                    [minput],
                    max_length=config.max_seq_length,
                    pad_to_max_length=True,
                    return_tensors="pt",
                    truncation=True, ## throws off warning if switched off
                )

                # tokenize targets
                tokenized_targets = tokenizer.batch_encode_plus(
                    [moutput],
                    max_length=config.max_answer,
                    pad_to_max_length=True,
                    return_tensors="pt",
                    truncation=True, ## throws off warning if switched off
                )
                if final_eval:
                    data_rep.append("%s\t%s\t%s" % (str(identifier),input_.replace("</s>",""),target.replace("</s>","")))

                ### print out some of the representations to spot check 
                if k <= 5 and (split == "train" or final_eval): # and dtype == 'train':
                    util_logger.info('************************')
                    util_logger.info('INPUT: %s' % minput)
                    util_logger.info('TARGET: %s' % moutput)
                    util_logger.info('INPUT TOKENS: %s' % ' '.join(input_tokens))
                    util_logger.info('OUTPUT TOKENS: %s' % ' '.join(output_tokens))

                inputs.append(tokenized_inputs)
                targets.append(tokenized_targets)

    ### 
    util_logger.info('Finished reading:inputs=%d, targets=%d' % (len(inputs),len(targets)))
    util_logger.info('Avg. input length: %d (max=%d, over=%d), Avg output length: %d (max=%d, over=%d)' %\
                         (np.mean(input_sizes),np.max(input_sizes),
                              len([s for s in input_sizes if s > config.max_seq_length]),
                              np.mean(output_sizes),np.max(output_sizes),
                              len([s for s in output_sizes if s > config.max_answer])))

    util_logger.info('length of data rep=%d' % len(data_rep))
    #return (inputs,targets,data_rep,original_sizes)
    return (inputs,targets,data_rep,output_sizes)

def json_patch(json_line):
    if "meta" in json_line and "prefix" in json_line["meta"]:
        json_line["prefix"] = json_line["meta"]["prefix"]
    if "answer" in json_line:
        json_line["output"] = json_line["answer"]
    if "context" in json_line:
        del json_line["question"]
        json_line["question"] = {}
        json_line["question"]["stem"] = json_line["context"]
        del json_line["context"]
        


def multi_qa(config,
                 tokenizer,
                 split,
                 final_eval):
    split_file = os.path.join(config.data_dir,"%s.jsonl" % split)
    util_logger.info('Reading file: %s, final eval=%s' % (split_file,final_eval))
    inputs   = []
    targets  = []
    data_rep = []
    input_sizes = []
    output_sizes = []

    with open(split_file,"r",encoding="utf-8") as f:
        for k,line in tqdm(enumerate(f)):
            json_line = json.loads(line.strip())

            ### 
            json_patch(json_line)
            
            identifier = "%s_%d" % (split,k) if "id" not in json_line else json_line["id"]

            # ### skip over explanation stuff 
            if (split == "dev" or split == "test") and not final_eval and json_line["prefix"] != "answer:": continue
            ## find the prefix
            if final_eval or split == "train": prefix = json_line["prefix"]
            else: prefix = "answer:"

            # # #### encoder side 
            #input_ = "%s %s </s>" % (prefix,json_line["question"]["stem"])
            input_ = "%s %s" % (prefix,json_line["question"]["stem"])
            input_ = re.sub(r'\s+|\n+',' ',input_)
            ## decoder side
            #target = "%s </s>" % json_line["output"]
            target = "%s" % json_line["output"]

            ##
            input_tokens = tokenizer.tokenize(input_)
            output_tokens = tokenizer.tokenize(target)
            input_sizes.append(len(input_tokens))
            output_sizes.append(len(output_tokens))

            tokenized_inputs = tokenizer.batch_encode_plus(
                    [input_],
                    max_length=config.max_seq_length,
                    #pad_to_max_length=True,
                    padding='max_length',
                    return_tensors="pt",
                    truncation=True, ## throws off warning if switched off
                )

            tokenized_targets = tokenizer.batch_encode_plus(
                    [target],
                    max_length=config.max_answer,
                    #pad_to_max_length=True,
                    padding='max_length',
                    return_tensors="pt",
                    truncation=True, ## throws off warning if switched off
                )

            if final_eval:
                data_rep.append("%s\t%s\t%s" % (str(identifier),input_.replace("</s>",""),target.replace("</s>","")))

            ### print out some of the representations to spot check 
            if k <= 5 and (split == "train" or final_eval): # and dtype == 'train':
                util_logger.info('************************')
                util_logger.info('INPUT: %s' % input_)
                util_logger.info('TARGET: %s' % target)
                util_logger.info('INPUT TOKENS: %s' % ' '.join(input_tokens))
                util_logger.info('OUTPUT TOKENS: %s' % ' '.join(output_tokens))

            inputs.append(tokenized_inputs)
            targets.append(tokenized_targets)

    util_logger.info('Finished reading:inputs=%d, targets=%d' % (len(inputs),len(targets)))
    util_logger.info('Avg. input length: %d (max=%d, over=%d), Avg output length: %d (max=%d, over=%d)' %\
                         (np.mean(input_sizes),np.max(input_sizes),
                              len([s for s in input_sizes if s > config.max_seq_length]),
                              np.mean(output_sizes),np.max(output_sizes),
                              len([s for s in output_sizes if s > config.max_answer])))

    util_logger.info('length of data rep=%d' % len(data_rep))
    return (inputs,targets,data_rep)


def json_mcqa(config,
                  tokenizer,
                  split,
                  final_eval):
    """Json mcqa extractor and feature builder

    :param config: the global configuration 
    :param tokenizer: the model tokenizer 
    :param split: the datasplit 
    :param full_generation: the type of generation needed
    :raises: ValuError
    """
    ####
    split_file = os.path.join(config.data_dir,"%s.jsonl" % split)
    main_prompt = "answer:" if not final_eval else "explain:"

    ## default switch
    if not config.split_explanations: main_prompt = "answer:"
    
    util_logger.info('Reading file: %s, final eval=%s, prompt=%s' % (split_file,final_eval,main_prompt))
    if not os.path.isfile(split_file):
        raise ValueError('Unknown file: %s' % split_file)

    inputs   = []
    targets  = []
    data_rep = []
    input_sizes = []
    output_sizes = []

    with open(split_file,"r",encoding="utf-8") as f:
        for k,line in enumerate(f):
            json_line = json.loads(line.strip())
            explanation = None
            identifier = "%s_%d" % (split,k) if "id" not in json_line else json_line["id"]
            if "para" in json_line: explanation = json_line["para"]

            ### 
            label=[k for k,c in enumerate(json_line["question"]["choices"]) \
                         if c["label"] == json_line["answerKey"]][0]
            endings = [c["text"] for c in json_line["question"]["choices"]]
            input_  = json_line["question"]["stem"]
            options = ['%s: %s' % (str(i+1), option) for i, option in enumerate(endings)]
            options = " ".join(options)

            ## formatting 
            input_ = "%s options: %s </s>" % (input_, options)

            ## normalize input here 
            input_ = re.sub(r'\s+|\n+',' ',input_)
            ### add explanation
            target = "%s </s>" % str(int(label) + 1)
            expl_target = None

            if explanation:
                expl_target = "[EXPL] %s </s>" % (re.sub(r'\n+',' ',re.sub(r'\s+',' ',explanation)))
            if final_eval:
                #prefix = "none" if "prefix" not in json_line else "none:"
                data_rep.append("%s\t%s\t%s" % (str(identifier),input_.replace("</s>",""),int(label)+1))
            if not config.split_explanations and explanation:
                target = "%s %s" % (str(int(label) + 1),expl_target)
                expl_target = None

            for minput,moutput in [("%s %s" % (main_prompt,input_),target),("explain: %s" % input_,expl_target)]:
                if moutput is None: continue
                input_tokens = tokenizer.tokenize(minput)
                output_tokens = tokenizer.tokenize(moutput)
                input_sizes.append(len(input_tokens))
                output_sizes.append(len(output_tokens))

                ## featurize
                tokenized_inputs = tokenizer.batch_encode_plus(
                    [minput],
                    max_length=config.max_seq_length,
                    pad_to_max_length=True,
                    return_tensors="pt",
                    truncation=True, ## throws off warning if switched off
                )

                # tokenize targets
                tokenized_targets = tokenizer.batch_encode_plus(
                    [moutput],
                    max_length=config.max_answer,
                    pad_to_max_length=True,
                    return_tensors="pt",
                    truncation=True, ## throws off warning if switched off
                )

                ### print out some of the representations to spot check 
                if k <= 5 and (split == "train" or final_eval): # and dtype == 'train':
                    util_logger.info('************************')
                    util_logger.info('INPUT: %s' % minput)
                    util_logger.info('TARGET: %s' % moutput)
                    util_logger.info('INPUT TOKENS: %s' % ' '.join(input_tokens))
                    util_logger.info('OUTPUT TOKENS: %s' % ' '.join(output_tokens))

                inputs.append(tokenized_inputs)
                targets.append(tokenized_targets)

    ### 
    util_logger.info('Finished reading:inputs=%d, targets=%d' % (len(inputs),len(targets)))
    util_logger.info('Avg. input length: %d (max=%d, over=%d), Avg output length: %d (max=%d, over=%d)' %\
                         (np.mean(input_sizes),np.max(input_sizes),
                              len([s for s in input_sizes if s > config.max_seq_length]),
                              np.mean(output_sizes),np.max(output_sizes),
                              len([s for s in output_sizes if s > config.max_answer])))

    util_logger.info('length of data rep=%d' % len(data_rep))
    return (inputs,targets,data_rep)


def single_token_eval(outputs,targets,final_eval=False):
    """Compares targets to first token in output (which should correspond to output token in 
    case of classification)

    :param outputs: the model (generated) outputs 
    :type outputs: list 
    :param targets: the model target
    :type targets: list 
    """
    ## log the first few
    util_logger.info('First few (outputs): %s' % ' '.join(outputs[:3]))
    util_logger.info('First few (targets): %s' % ' '.join(targets[:3]))
    
    ## update for new tokenizer 
    new_outputs = [o.replace("<pad>","").replace("</s>","").strip().split()[0]  if o.strip() else "" for o in outputs]
    new_targets = [t.replace("<pad>","").replace("</s>","").strip() for t in targets]

    util_logger.info('First few (outputs, processed): %s' % ' '.join(new_outputs[:3]))
    util_logger.info('First few (targets, processed): %s' % ' '.join(new_targets[:3]))
    #score = sklearn_metrics.accuracy_score(targets, new_outputs)
    score = sklearn_metrics.accuracy_score(new_targets, new_outputs)
    util_logger.info('resulting score: %f, length of inputs=%d' % (score,len(new_outputs)))
    #return torch.from_numpy(np.array(score)).double()
    return score

def bleu_eval(outputs,targets,final_eval=False):
    """Compares targets to first token in output (which should correspond to output token in 
    case of classification)

    :param outputs: the model (generated) outputs 
    :type outputs: list 
    :param targets: the model target
    :type targets: list 
    """
    # new_outputs = [[o.strip()] for o in outputs]
    # weights = [1/len(targets)] * len(targets)
    # score = corpus_bleu(new_outputs,targets,weights=weights,smoothing_function=SmoothingFunction().method1)
    # util_logger.info('Resulting bleu score: %f, length of inputs=%d' % (score,len(new_outputs)))
    
    bl_scores = []
    for k,output in enumerate(outputs):
        score = bleu([output],targets[k],smoothing_function=SmoothingFunction().method1)
        bl_scores.append(score)
    return np.mean(score)

def print_full_with_bleu(outputs,targets,data_rep,ofile):
    bl_scores = []
    with open(ofile,'w') as output_file:
        for k,output in enumerate(outputs):
            score = bleu([output],targets[k],smoothing_function=SmoothingFunction().method1)
            bl_scores.append(score)
            print("%s\t%s\t%s\t%f" % (data_rep[k],targets[k],output,score),file=output_file)
    util_logger.info("Final bleu score on final eval: %f" % np.mean(bl_scores))

def full_seq_eval(output,targets,data_rep,ofile):
    output = [["B-%s" % w for w in o.split()] for o in output]
    targets = [["B-%s" % w for w in o.split()] for o in targets]

    out_directory = os.path.dirname(ofile)
    report = os.path.join(out_directory,"classification_report.txt")
    with open(report,'w') as class_report:
        print(classification_report(targets,output),file=class_report)
    
            
def print_full_output(outputs,targets,data_rep,ofile,print_bleu=False):
    """Print full model output 

    :param outputs: the output of the model 
    :param targets: the gold targets 
    :param data_rep: the representation of the original data
    :param ofile: the model output file 
    :param print_bleu: print bleu score (in case of generation)
    """
    util_logger.info('Printing model output to: %s' % ofile)
    util_logger.info('Portion of output: %s' % ', '.join(outputs[:5]))
    util_logger.info('Portion of targets: %s' % ', '.join(targets[:5]))

    ### 
    with open(ofile,'w') as output_file:
        for k,output in enumerate(outputs):
            #print("%s\t%s\t%s" % (data_rep[k],targets[k],output),file=output_file)
            print("%s\t%s" % (data_rep[k],output.replace("<pad>","").replace("</s>","").strip()),file=output_file)

    ### print bleu scores for generation outputs
    if print_bleu: 
        generation_out = os.path.join(os.path.dirname(ofile),"generation_eval.tsv")
        with open(generation_out,'w') as gen_out:
            for k,output in enumerate(outputs):
                score = bleu([output],targets[k],smoothing_function=SmoothingFunction().method1)
                print("%s\t%s\t%f" % (data_rep[k],output,score),file=gen_out)
                
def prepare_explanations(config,tokenizer):
    """This is a bit of a hack, but sets up the mcqa format to get out the explanations. 
    
    :param config: the global configuration (i.e., used the t5 model and trainers)
    :param tokenizer: the model tokenizer (for computing max/min lengths)
    :returns: list 
    """
    special_tokens = set()
    maximum_size = 0
    
    new_train = os.path.join(config.output_dir,"train.jsonl")
    with open(os.path.join(config.data_dir,"train.jsonl")) as training_data:
        with open(new_train,'w') as new_train:
            for k,line in enumerate(training_data):
                json_line = json.loads(line)
                answer_label = json_line["answerKey"]
                explanation_list = []
                running_size = 0

                for choice in json_line["question"]["choices"]:

                    ## remove tokens
                    if "tokens" in choice: del choice["tokens"]
                    if "query" in choice: del choice["query"]
                    if choice["label"] == answer_label:
                        if "para" not in choice:
                                logging.warning('Training item #%d missing gold IR, skipping..' % k)
                                continue
                        para = choice["para"]
                        para_lf = choice["para_lf"]
                        para = [f for f in para.split(". ") if f.strip()]
                        if len(para_lf) != len(para): continue
                        already = set()

                        ##
                        for num_facts,(lf_info,explanation) in enumerate(zip(para_lf,para)):
                            if num_facts >= (config.num_facts-1): break 
                            head_word,lf = lf_info
                            if explanation in already: continue
                            already.add(explanation)
                            knowledge_type = "[%s]" % lf.split()[0].split('.')[-1].strip().upper()
                            special_tokens.add(knowledge_type)
                            explanation = "%s %s: %s" % (knowledge_type,head_word,explanation)
                            llength = len([t for t in tokenizer.tokenize(explanation)])
                            if (running_size + llength + 3) > config.max_explanation: continue
                            running_size += llength
                            explanation_list.append(explanation)

                        ## remove some of the fields
                        del choice["para"]
                        del choice["para_lf"]

                    ## remove fields for other stuff 
                    else:
                        if "para" in choice: del choice["para"]
                        if "para_lf" in choice: del choice["para_lf"]

                ## add final explanation
                json_line["para"] = ' '.join(explanation_list)
                if running_size > maximum_size:
                    maximum_size = running_size
                new_train.write(json.dumps(json_line))
                new_train.write("\n")

        ## go through dev
        new_dev = os.path.join(config.output_dir,"dev.jsonl")
        with open(os.path.join(config.data_dir,"dev.jsonl")) as dev:
            with open(new_dev,'w') as new_dev: 
                for line in dev:
                    json_line = json.loads(line)
                    for choice in json_line["question"]["choices"]:
                        if "para" in choice: del choice["para"]
                        if "para_lf" in choice: del choice["para_lf"]
                        if "tokens" in choice: del choice["tokens"]
                        ###

                    new_dev.write(json.dumps(json_line))
                    new_dev.write("\n")

        ## go through test (if it exists, merge code with code above)
        orig_test = os.path.join(config.data_dir,"test.jsonl")
        if os.path.isfile(orig_test):
            new_dev = os.path.join(config.output_dir,"test.jsonl")
            with open(os.path.join(config.data_dir,"test.jsonl")) as dev:
                with open(new_dev,'w') as new_dev: 
                    for line in dev:
                        json_line = json.loads(line)
                        for choice in json_line["question"]["choices"]:
                            if "para" in choice: del choice["para"]
                            if "para_lf" in choice: del choice["para_lf"]
                            if "tokens" in choice: del choice["tokens"]
                        ###
                        new_dev.write(json.dumps(json_line))
                        new_dev.write("\n")
        ##
        config.max_answer = maximum_size + 3
        config.data_dir = config.output_dir
        util_logger.info('Maximum answer size after formatting=%d, new data_dir=%s, # special characters=%d' %\
                             (config.max_answer,config.data_dir,len(special_tokens)))
        ### return the special tokens 
        return list(special_tokens)

def prepare_multi(config,tokenizer):
    """Extract the special tokens from the multiQA data 

    """
    special_tokens = ["[FACT]","[FACT_PATTERN]","[CONCLUSIONS]"]
    
    with open(os.path.join(config.data_dir,"train.jsonl")) as training_data:
        for k,line in enumerate(training_data):
            json_line = json.loads(line)
            if "output" not in json_line: continue 
            special_tokens +=  re.findall(r'\[[A-Z\_\-]+\]',str(json_line["output"]))

            special_tokens += re.findall(r'rel\=([A-Z\-\_]+)',str(json_line["output"]))
            if "stem" in json_line["question"]: 
                special_tokens += re.findall(r'rel\=([A-Z\-\_]+)',str(json_line["question"]["stem"]))

    util_logger.info('found %d special tokens' % len(set(special_tokens)))
    return list(set([s.strip() for s in special_tokens]))

def multi_query(text_input,tokenizer,config,prefix):
    """Process the text query to run through the model 
    
    :param text_input" the text to prepare
    :param prefix: specifies the model mode
    """
    input_ = "%s %s" % (prefix,text_input)
    input_ = re.sub(r'\s+|\n+',' ',input_)
    target = "?"

    tokenized_inputs = tokenizer.batch_encode_plus(
                    [input_],
                    max_length=config.max_seq_length,
                    #pad_to_max_length=True,
                    return_tensors="pt",
                    truncation=True, ## throws off warning if switched off
                )
    tokenized_targets = tokenizer.batch_encode_plus(
                    [target],
                    max_length=config.max_answer,
                    #pad_to_max_length=True,
                    return_tensors="pt",
                    truncation=True, ## throws off warning if switched off
                )
    return ([tokenized_inputs],[tokenized_targets])
