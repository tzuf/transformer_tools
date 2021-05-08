# Demo for T5 trained on multi-angular AAAC

# Core Pkgs
import streamlit as st 
import os
import sys
import re
import pandas as pd
sys.path.append('.')
from optparse import OptionParser,OptionGroup
from transformer_tools.T5Classification import params as tparams
from transformer_tools import initialize_config
from transformer_tools import LoadT5Classifier
from transformer_tools.util.cache import LRUCache


BOILER_PLATE = {
    "Enter your story" : "",
    "Enter question (optional)": "",
}

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

TEST_DATA = [
    {"argument_source":"A person who is not a nephew of Richard is a half-brother of Lance or a son of Jeff, and vice versa. Hence, somebody who is not a nephew of Richard is a half-brother of Lance or a son of Jeff. We may conclude that nobody is neither a nephew of Richard nor a son of Jeff. All this entails that a person who is not a great-grandfather of David is a son of Jeff, owing to the fact that someone who is not a great-grandfather of David is not a nephew of Richard.","argdown_reconstruction":"(1) If, and only if, someone is not a nephew of Richard, then they are a half-brother of Lance or a son of Jeff.\n--\nwith generalized biconditional elimination (negation variant, complex variant) from (1)}\n--\n(2) If someone is not a nephew of Richard, then they are a half-brother of Lance or a son of Jeff.\n(3) If someone is a half-brother of Lance, then they are a nephew of Richard.\n--\nwith generalized disjunctive syllogism (transposition, negation variant) from (2), (3)}\n--\n(4) If someone is not a nephew of Richard, then they are a son of Jeff.\n(5) If someone is not a great-grandfather of David, then they are not a nephew of Richard.\n--\nwith hypothetical syllogism (negation variant) from (4), (5)}\n--\n(6) If someone is not a great-grandfather of David, then they are a son of Jeff.","reason_statements":[{"text":"A person who is not a nephew of Richard is a half-brother of Lance or a son of Jeff, and vice versa","starts_at":0,"ref_reco":1},{"text":"someone who is not a great-grandfather of David is not a nephew of Richard","starts_at":383,"ref_reco":5}],"conclusion_statements":[{"text":"somebody who is not a nephew of Richard is a half-brother of Lance or a son of Jeff","starts_at":108,"ref_reco":2},{"text":"nobody is neither a nephew of Richard nor a son of Jeff","starts_at":214,"ref_reco":4},{"text":"a person who is not a great-grandfather of David is a son of Jeff","starts_at":293,"ref_reco":6}],"premises":[{"ref_reco":1,"text":"If, and only if, someone is not a nephew of Richard, then they are a half-brother of Lance or a son of Jeff.","explicit":"true"},{"ref_reco":3,"text":"If someone is a half-brother of Lance, then they are a nephew of Richard.","explicit":"false"},{"ref_reco":5,"text":"If someone is not a great-grandfather of David, then they are not a nephew of Richard.","explicit":"true"}],"premises_formalized":[{"form":"(x): \u00ac${F2}x <-> (${F4}x v ${F3}x)","ref_reco":1},{"form":"(x): ${F4}x -> ${F2}x","ref_reco":3},{"form":"(x): \u00ac${F1}x -> \u00ac${F2}x","ref_reco":5}],"conclusion":[{"ref_reco":6,"text":"If someone is not a great-grandfather of David, then they are a son of Jeff."}],"conclusion_formalized":[{"form":"(x): \u00ac${F1}x -> ${F3}x","ref_reco":6}],"intermediary_conclusions_formalized":[{"form":"(x): \u00ac${F2}x -> (${F4}x v ${F3}x)","ref_reco":2},{"form":"(x): \u00ac${F2}x -> ${F3}x","ref_reco":4}],"intermediary_conclusions":[{"ref_reco":2,"text":"If someone is not a nephew of Richard, then they are a half-brother of Lance or a son of Jeff."},{"ref_reco":4,"text":"If someone is not a nephew of Richard, then they are a son of Jeff."}],"distractors":[],"id":"5812d3a0-05d0-4e50-af62-416205f6ea22","predicate_placeholders":["F1","F2","F3","F4"],"entity_placeholders":[],"steps":3,"n_premises":3,"n_distractors":0,"base_scheme_groups":["hypothetical syllogism","generalized biconditional elimination","generalized disjunctive syllogism"],"scheme_variants":["transposition","negation variant","complex variant"],"domain_id":"male_relatives","domain_type":"persons","plcd_subs":{"F1":"great-grandfather of David","F2":"nephew of Richard","F3":"son of Jeff","F4":"half-brother of Lance"},"argdown_index_map":{"s0c":6,"s0p0":5,"s1c":4,"s1p1":3,"s2c":2,"s2p0":1},"presentation_parameters":{"resolve_steps":[],"direction":"forward","implicit_conclusion":"false","implicit_premise":"true","redundancy_frequency":0.1,"drop_conj_frequency":0.1,"start_sentence":[0,1]}},
    {"argument_source":"Whoever is a sufferer of allergy to mango is not a sufferer of allergy to sesame or a sufferer of allergy to carrot. And no sufferer of allergy to carrot is hypersensitive to mango. Consequently, every sufferer of allergy to mango reacts allergically to turkey. Yet someone who is not a sufferer of allergy to mango and a sufferer of allergy to cheese is a sufferer of allergy to ginger and a sufferer of allergy to pepper. Plus, every person who is not both a sufferer of allergy to maize and a sufferer of allergy to mustard is a sufferer of allergy to cinnamon or a sufferer of allergy to oat.","argdown_reconstruction":"(1) If someone is a sufferer of allergy to mango, then they are a sufferer of allergy to carrot, or not a sufferer of allergy to sesame.\n(2) If someone is a sufferer of allergy to carrot, then they are not a sufferer of allergy to mango.\n--\nwith generalized disjunctive syllogism {variant: [transposition, negation variant]; uses: (1), (2)}\n--\n(3) If someone is a sufferer of allergy to mango, then they are not a sufferer of allergy to sesame.\n(4) If someone is a sufferer of allergy to mango, then they are a sufferer of allergy to sesame or a sufferer of allergy to turkey.\n--\nwith generalized disjunctive syllogism {uses: (3), (4)}\n--\n(5) If someone is a sufferer of allergy to mango, then they are a sufferer of allergy to turkey.","reason_statements":[{"text":"Whoever is a sufferer of allergy to mango is not a sufferer of allergy to sesame or a sufferer of allergy to carrot","starts_at":0,"ref_reco":1},{"text":"no sufferer of allergy to carrot is hypersensitive to mango","starts_at":121,"ref_reco":2}],"conclusion_statements":[{"text":"every sufferer of allergy to mango reacts allergically to turkey","starts_at":196,"ref_reco":5}],"premises":[{"ref_reco":1,"text":"If someone is a sufferer of allergy to mango, then they are a sufferer of allergy to carrot, or not a sufferer of allergy to sesame.","explicit":"true"},{"ref_reco":2,"text":"If someone is a sufferer of allergy to carrot, then they are not a sufferer of allergy to mango.","explicit":"true"},{"ref_reco":4,"text":"If someone is a sufferer of allergy to mango, then they are a sufferer of allergy to sesame or a sufferer of allergy to turkey.","explicit":"false"}],"premises_formalized":[{"form":"(x): ${F1}x -> (${F4}x v \u00ac${F2}x)","ref_reco":1},{"form":"(x): ${F4}x -> \u00ac${F1}x","ref_reco":2},{"form":"(x): ${F1}x -> (${F2}x v ${F3}x)","ref_reco":4}],"conclusion":[{"ref_reco":5,"text":"If someone is a sufferer of allergy to mango, then they are a sufferer of allergy to turkey."}],"conclusion_formalized":[{"form":"(x): ${F1}x -> ${F3}x","ref_reco":5}],"intermediary_conclusions_formalized":[{"form":"(x): ${F1}x -> \u00ac${F2}x","ref_reco":3}],"intermediary_conclusions":[{"ref_reco":3,"text":"If someone is a sufferer of allergy to mango, then they are not a sufferer of allergy to sesame."}],"distractors":["Every person who is not both a sufferer of allergy to maize and a sufferer of allergy to mustard is a sufferer of allergy to cinnamon or a sufferer of allergy to oat.","Someone who is not a sufferer of allergy to mango and a sufferer of allergy to cheese is a sufferer of allergy to ginger and a sufferer of allergy to pepper."],"id":"8c2c3329-cab8-4bd1-b4e7-3ff26506be9d","predicate_placeholders":["F1","F2","F3","F4"],"entity_placeholders":[],"steps":2,"n_premises":3,"n_distractors":2,"base_scheme_groups":["generalized disjunctive syllogism"],"scheme_variants":["transposition","negation variant"],"domain_id":"allergies","domain_type":"persons","plcd_subs":{"F1":"sufferer of allergy to mango","F2":"sufferer of allergy to sesame","F3":"sufferer of allergy to turkey","F4":"sufferer of allergy to carrot"},"argdown_index_map":{"s0c":5,"s0p0":4,"s1c":3,"s1p1":2,"s2c":2,"s2p0":1,"s1p0":1},"presentation_parameters":{"resolve_steps":[1],"direction":"forward","implicit_conclusion":"false","implicit_premise":"true","redundancy_frequency":0.1,"drop_conj_frequency":0.1,"start_sentence":[0,2]}},
    {"argument_source":"If, and only if, someone is a critic of Besiktas JK, then they are an expert of Tottenham Hotspur or a friend of KRC Genk. And everybody who hasn't expert knowledge about Kilmarnock FC doesn't criticize Besiktas JK, and no expert of Kilmarnock FC has expert knowledge about Tottenham Hotspur, and vice versa.","argdown_reconstruction":"(1) If, and only if, someone is a critic of Besiktas JK, then they are an expert of Tottenham Hotspur or a friend of KRC Genk.\n--\nwith generalized biconditional elimination (negation variant, complex variant) from (1)}\n--\n(2) If someone is a critic of Besiktas JK, then they are an expert of Tottenham Hotspur or a friend of KRC Genk.\n(3) If, and only if, someone is an expert of Kilmarnock FC, then they are not an expert of Tottenham Hotspur.\n--\nwith generalized biconditional elimination (negation variant) from (3)}\n--\n(4) If someone is an expert of Kilmarnock FC, then they are not an expert of Tottenham Hotspur.\n(5) If someone is not an expert of Kilmarnock FC, then they are not a critic of Besiktas JK.\n--\nwith hypothetical syllogism (transposition, negation variant) from (4), (5)}\n--\n(6) If someone is a critic of Besiktas JK, then they are not an expert of Tottenham Hotspur.\n--\nwith generalized disjunctive syllogism from (2), (6)\n--\n(7) If someone is a critic of Besiktas JK, then they are a friend of KRC Genk.","reason_statements":[{"text":"If, and only if, someone is a critic of Besiktas JK, then they are an expert of Tottenham Hotspur or a friend of KRC Genk","starts_at":0,"ref_reco":1},{"text":"everybody who hasn't expert knowledge about Kilmarnock FC doesn't criticize Besiktas JK","starts_at":127,"ref_reco":5},{"text":"no expert of Kilmarnock FC has expert knowledge about Tottenham Hotspur, and vice versa","starts_at":220,"ref_reco":3}],"conclusion_statements":[],"premises":[{"ref_reco":1,"text":"If, and only if, someone is a critic of Besiktas JK, then they are an expert of Tottenham Hotspur or a friend of KRC Genk.","explicit":"true"},{"ref_reco":3,"text":"If, and only if, someone is an expert of Kilmarnock FC, then they are not an expert of Tottenham Hotspur.","explicit":"true"},{"ref_reco":5,"text":"If someone is not an expert of Kilmarnock FC, then they are not a critic of Besiktas JK.","explicit":"true"}],"premises_formalized":[{"form":"(x): ${F1}x <-> (${F2}x v ${F3}x)","ref_reco":1},{"form":"(x): ${F4}x <-> \u00ac${F2}x","ref_reco":3},{"form":"(x): \u00ac${F4}x -> \u00ac${F1}x","ref_reco":5}],"conclusion":[{"ref_reco":7,"text":"If someone is a critic of Besiktas JK, then they are a friend of KRC Genk."}],"conclusion_formalized":[{"form":"(x): ${F1}x -> ${F3}x","ref_reco":7}],"intermediary_conclusions_formalized":[{"form":"(x): ${F1}x -> (${F2}x v ${F3}x)","ref_reco":2},{"form":"(x): ${F4}x -> \u00ac${F2}x","ref_reco":4},{"form":"(x): ${F1}x -> \u00ac${F2}x","ref_reco":6}],"intermediary_conclusions":[{"ref_reco":2,"text":"If someone is a critic of Besiktas JK, then they are an expert of Tottenham Hotspur or a friend of KRC Genk."},{"ref_reco":4,"text":"If someone is an expert of Kilmarnock FC, then they are not an expert of Tottenham Hotspur."},{"ref_reco":6,"text":"If someone is a critic of Besiktas JK, then they are not an expert of Tottenham Hotspur."}],"distractors":[],"id":"ead34d89-af68-4add-bb62-caff9043c90f","predicate_placeholders":["F1","F2","F3","F4"],"entity_placeholders":[],"steps":4,"n_premises":3,"n_distractors":0,"base_scheme_groups":["hypothetical syllogism","generalized biconditional elimination","generalized disjunctive syllogism"],"scheme_variants":["transposition","negation variant","complex variant"],"domain_id":"football_fans","domain_type":"persons","plcd_subs":{"F1":"critic of Besiktas JK","F2":"expert of Tottenham Hotspur","F3":"friend of KRC Genk","F4":"expert of Kilmarnock FC"},"argdown_index_map":{"s0c":7,"s1c":2,"s1p1":2,"s2c":6,"s2p0":5,"s1p0":1,"s3c":4,"s3p0":3},"presentation_parameters":{"resolve_steps":[1,2,3],"direction":"backward","implicit_conclusion":"true","implicit_premise":"false","redundancy_frequency":0.1,"drop_conj_frequency":0.1,"start_sentence":[0,2]}}
]

CACHE_SIZE = 10000

def params(config):
    """Main parameters for running the T5 model

    :param config: the global configuration object
    """
    tparams(config)

    group = OptionGroup(config,"transformer_tools.Tagger",
                            "Settings for tagger models")

    config.add_option_group(group)

    
@st.cache(allow_output_mutation=True)
def build_config():
    config = initialize_config(sys.argv[1:],params)
    config.T5_type = 'T5ClassificationMultiQA'
    config.max_answer = 300
    config.max_seq_len = 300
    config.no_repeat_ngram_size = 0
    return config


@st.cache
def build_model(config):
    model = LoadT5Classifier(config)
    return model

@st.cache(allow_output_mutation=True)
def get_cache():
    cache = LRUCache(CACHE_SIZE)
    return cache
    


@st.cache(allow_output_mutation=True)
def aaac_fields():
    fields = []
    for m in MODES:
        fields = fields + m['from']
    return set(sorted(fields))

# defines how to present reason and conclusion statements to the model
@st.cache(allow_output_mutation=True)
def format_statements_list(statements: list) -> str:
    if len(statements)==0:
        return "None"
    list_as_string = ["%s (ref: (%s))" % (sdict['text'],sdict['ref_reco']) for sdict in statements]
    list_as_string = " | ".join(list_as_string)
    return list_as_string

# format raw argdown (inserting line breaks)
@st.cache(allow_output_mutation=True)
def format_argdown(raw_argdown: str) -> str:
    format_statement_block = lambda s: re.sub('(\([0-9]+\))', r'<br>\1', s)
    format_inference_block = lambda s: "<br>--<br><i>"+s+"</i><br>--"
    split = raw_argdown.split(' -- ')
    argdown = format_statement_block(split[0])
    i=1
    while i<len(split):
        argdown = argdown + format_inference_block(split[i])
        argdown = argdown + format_statement_block(split[i+1])
        i = i+2
    argdown = argdown[3:]# remove first linebreak
    return argdown



def run_model(mode_set, user_input):
    """Main method for running the model 

    :param mode_set: the modes to run
    :param user_input: the user input (dict) 
    :returns: output dict
    """
    config = build_config()
    model = build_model(config)
    cache = get_cache()

    current_input = user_input.copy()
    output = []

    for i,mode_id in enumerate(mode_set):
        current_mode = next(m for m in MODES if m['id']==mode_id)
        # construct prompt
        inquire_prompt = ""
        for from_key in current_mode['from']:
            inquire_prompt = inquire_prompt + ("%s: %s " % (from_key,current_input[from_key]))
        to_key = current_mode['to']
        #inquire_prompt = inquire_prompt + to_key + ":" # comment out this line if custom prefix used
        # inquire model
        out = model.query(inquire_prompt,prefix=to_key+":")[0] # change this to: to_key+":" rather than "gen:"
        # write output
        output.append({
            'step':i,
            'mode':current_mode,
            'output':out,
            'prompt':inquire_prompt
        })
        # update input
        current_input[to_key] = out

    return output


            

def main():

    #config = build_config()
    #model = build_model(config)
    st.title("T5 AAAC Interface")

    ## page details

    # choose example data
    ex_texts = [x['argument_source'] for x in TEST_DATA]
    ex_texts = ['...'] + ex_texts
    ex_s = st.selectbox("Select example:",ex_texts,index=0)
    ex_item = TEST_DATA[ex_texts.index(ex_s)-1]

    user_input = {}

    # for every mode, add input field
    for d in aaac_fields():
        filler = d
        if ex_s != '...':
            filler = ex_item[d]
            if d in ['reason_statements','conclusion_statements']:
                filler = format_statements_list(filler)

        user_input[d] = st.text_area(
            d,filler,
            height=250
        )

        if user_input[d]==d:
            user_input[d] = ""


    #story + answer + supporting facts => question 
    modes_s = st.multiselect(
        "Modular Computations (argument source=`s`, reason statements=`r`, conclusion statements=`c`, argdown recunstruction=`a`)",
        [m['id'] for m in MODES],
        ["s => a"]
    )

    ## answer a query
    submit = st.button("Process")
    row = []; index = []
    row_der = []; index_der = []


    if submit:
        with st.spinner("Processing..."):
            output = run_model(modes_s,user_input)
            st.write("<b> Generated output </b>",unsafe_allow_html=True)
            for out in output:
                st.write("step: %d, mode: %s" % (out['step'],out['mode']['id']))
                output_f = format_argdown(out['output']) if out['mode']['to']=='argdown_reconstruction' else out['output']
                st.write(output_f, unsafe_allow_html=True)

        
if __name__ == '__main__':
    main()
