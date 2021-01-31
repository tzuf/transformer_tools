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
    config.max_answer = 100
    config.max_seq_len = 300
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

    #story + answer + supporting facts => question 
    modes = st.multiselect(
        "Modular Computations (story=`s`, question=`q`, support=`sp`, answer=`a`)",
        ["s+q => a","s+q => sp","sp+s+q => a","s+a+sp => q"],
        ["s+q => a"]
    )
    
    ## answer a query
    submit = st.button("Process")
    row = []; index = []
    row_der = []; index_der = []

    if story_text and question and submit:
        mode_set = set(modes)
        ### s+q => a
        q_input = "%s $question$ %s" % (story_text,question)
        model_out = model.query(q_input)
        answer_text = model_out[0]
        row.append([q_input,answer_text])
        index.append("s+q => a")

        ## s+q => sp
        if "s+q => sp" in mode_set or "sp+s+q => a" in mode_set or "s+a+sp => q" in mode_set:
            supporting_text = "$question$ %s $story$ %s" % (question,story_text)
            support_out = model.query(supporting_text,prefix="generate:")[0]
            row.append([supporting_text,support_out])
            index.append("s+q => sp")

            if "sp+s+q => a" in mode_set:
                full_input = "$context$ %s $story$ %s $question$ %s" % (support_out,story_text,question)
                second_answer_out = model.query(full_input,prefix="answer:")[0]
                row_der.append([full_input,second_answer_out])
                index_der.append("sp+s+q => a")

            if "s+a+sp => q" in mode_set:
                question_input = "$answer$ %s $context$ %s $story$ %s" % (answer_text,support_out,story_text)
                question_out = model.query(question_input,prefix="question:")[0]
                row_der.append([question_input,question_out])
                index_der.append("s+a+sp => q")

        ## explicit results 
        df = pd.DataFrame(
            row,
            columns=["input","output (predicted)"],
            index=index
        )
        st.write("<b> Direct Computations </b>",unsafe_allow_html=True)
        st.table(df)

        if index_der and row_der:
            st.write("<b> Derived Computations </b>",unsafe_allow_html=True)
            df2 = pd.DataFrame(
                row_der,
                columns=["input (predicted)","output (predicted)"],
                index=index_der
            )
            st.table(df2)
            
        
        
if __name__ == '__main__':
    main()
