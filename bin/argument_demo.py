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
def example_sets():
    return (
    [
    "...",
    "If, and only if, Keith is a expert of FC Vaduz and Keith is a member of FC Spartak Trnava, then Keith is a critic of FK Jablonec. In consequence, if Keith is a expert of FC Vaduz and Keith is a member of FC Spartak Trnava, then Keith is a critic of FK Jablonec. Plus, Keith is a expert of FC Vaduz and Keith is a member of FC Spartak Trnava, because Keith is not a expert of PSV Eindhoven or Keith is a critic of OGC Nice. Still, if it is not the case that Keith is a expert of FC Vaduz and Keith is a member of FC Spartak Trnava, then Keith is not a critic of OGC Nice. But if Keith is a friend of RC Celta de Vigo, then Keith is a expert of PSV Eindhoven, and if it is not the case that Keith is a expert of FC Vaduz and Keith is a member of FC Spartak Trnava, then Keith is not a critic of OGC Nice.",
    "Everyone who is a ancestor of Heather is a niece of Penny, too. As, every ancestor of Heather is a schoolmate of Katherine or a granddaughter of Della. In addition, if, and only if, someone is a schoolmate of Katherine, then they are a niece of Penny. Still, if it is not the case that someone is a ancestor of Janel or a niece of Lisa, then they are not a granddaughter of Della. But whoever is a ancestor of Janel or a niece of Lisa is a niece of Penny.",
    "Coal Tar is a ingredient of Off Duty, since Coal Tar is a ingredient of Face up To It or Coal Tar is a ingredient of Skin Cleanser, and if Coal Tar is a ingredient of Face up To It, then Coal Tar is a ingredient of Off Duty, because if Coal Tar is a ingredient of Face up To It, then Coal Tar is a ingredient of Laced Up. In addition, if Coal Tar is a ingredient of Laced Up, then Coal Tar is a ingredient of Off Duty. Which derives from the fact that everything that is a ingredient of Laced Up is a ingredient of Off Duty, too."
    ])

def argdown_structure(raw_output):
    name_column = []
    output_items = []
    current = 1
    
    for k,item in enumerate(re.split(r'\([0-9]+\) ',raw_output)):
        if not item: continue 
        first = "%s" % (item)

        if '--' in item:
            first = first.split(" --")[0]
            rule = re.search(r'(\-\- with .+)$',item).groups()[0]
            output_items.append(first)
            output_items.append(rule)
            name_column.append("(%d)" % current)
            name_column.append("rule")
        else: 
            output_items.append(first)
            name_column.append("(%d)" % current)
        current += 1

    df = pd.DataFrame(
        list(zip(name_column,output_items)),
        columns=["part","output (predicted)"],
        index=["" for i in range(len(name_column))],
    )
    return df

def run_model(mode_set,
                  argument_text):
    """Main method for running the model 

    :param mode_set: the modes to run
    :param argument_text: the main argument text 
    :returns: some dataframes to write out
    """
    config = build_config()
    model = build_model(config)
    cache = get_cache()

    ## for building data frames
    row = []; index = []
    row_der = []; index_der = []

    q_input = "%s" % argument_text ##<-- should be just this, no special rendering
    answer_text = cache[q_input]
    if answer_text is None:
        answer_text = model.query(q_input,prefix="gen:")[0]
        cache[q_input] = answer_text

    return argdown_structure(answer_text)
    

def main():
    st.title("Artificial Argument Analysis Demo")
    ex_arguments = example_sets()

    ex_s = st.selectbox("Select example:",ex_arguments,index=0)
    argument_filler = "Enter your argument"
    if ex_s != '...':
        argument_filler = ex_s
        q_filler = ex_arguments[ex_arguments.index(ex_s)]

    story_text = st.text_area(
        "Argument text",argument_filler,
        height=400
    )

    argument_text = BOILER_PLATE.get(story_text,story_text)

    modes = st.multiselect(
        "Modular Computations (argument=`arg_src`, argdown format=`argdown`, conclusion identifiers=`c`)",
        ["arg_src => argdown","argdown => c"],
        ["arg_src => argdown"]
    )

    submit = st.button("Process")
    row = []; index = []
    row_der = []; index_der = []

    if argument_text and submit:
        mode_set = set(modes)
        with st.spinner("Processing..."):
            answer_df = run_model(mode_set,argument_text)

            st.write("<b> Argdown generation </b>",unsafe_allow_html=True)
            st.table(answer_df)
            

if __name__ == "__main__":
    main() 
