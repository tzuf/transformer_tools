# Core Pkgs
import streamlit as st 
import os
import sys
import pandas as pd
sys.path.append('.')
from optparse import OptionParser,OptionGroup
from transformer_tools.T5Classification import params as tparams
from transformer_tools import initialize_config
from transformer_tools import LoadT5Classifier


BOILER_PLATE = {
    "Enter your story" : "",
    "Enter question (optional)": "",
}

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
    return config

@st.cache
def build_model(config):
    model = LoadT5Classifier(config)
    return model

def main():
    config = build_config()
    model = build_model(config)
    
    st.title("T5 bAbi Interface")

    choice = "story + question => answer"

    story_text = st.text_area(
        "story text","Enter your story",
        height=300
    )
    story_text = BOILER_PLATE.get(story_text,story_text)

    question = st.text_area(
        "other input: for a single question, just type `question`.","Enter question (optional)",
        height=1
    )
    question = BOILER_PLATE.get(question,question)

    modes = st.multiselect(
        "Modular Computations",["story + question => answer"],
        ["story + question => answer"]
    )
    
    ## answer a query
    submit = st.button("Process")

    if story_text and question and submit:
        q_input = "%s $question$ %s" % (story_text,question)
        model_out = model.query(q_input)

        df = pd.DataFrame(
            [[q_input,model_out[0]]],
            columns=["input","output (predicted)"],
            index=[choice]
       )

        st.table(df)
        
if __name__ == '__main__':
    main()
