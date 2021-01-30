# Core Pkgs
import streamlit as st 
import os
import sys
sys.path.append('../')
from transformer_tools import initialize_config

BOILER_PLATE = {
    "Enter your story" : "",
    "Enter question (optional)": "",
}


def main():
    st.title("T5 bAbi Interface")

    mode = ["story + question => answer"]
    choice = st.sidebar.selectbox("Model-mode",mode)

    story_text = st.text_area(
        "story text","Enter your story",
        height=300
    )
    story_text = BOILER_PLATE.get(story_text,story_text)

    question = st.text_area(
        "question","Enter question (optional)",
        height=1
    )
    question = BOILER_PLATE.get(question,question)

    ## answer a query
    submit = st.button("Process")

    if story_text and question and submit:
        st.write("<b> <tt> mode: </tt> </b> %s" % choice,unsafe_allow_html=True)
        st.write("<b> <tt> output: </tt> </b> %s" % story_text,unsafe_allow_html=True)

def params(config):
    """Main parameters for running the T5 model

    :param config: the global configuration object
    """
    #from transformer_tools.T5Classifier import params as tparams
    #tparams(config)

    group = OptionGroup(config,"transformer_tools.Tagger",
                            "Settings for tagger models")

    # group.add_option("--model_type",
    #                      dest="model_type",
    #                      default='bert-base-uncased',
    #                      type=str,
    #                      help="The type of tagger to use [default='bert-base-cased']")

    # group.add_option("--model_name",
    #                      dest="model_name",
    #                      default='bert',
    #                      type=str,
    #                      help="The name of the model [default='bert']")

    # group.add_option("--tagger_model",
    #                      dest="tagger_model",
    #                      default='arrow_tagger',
    #                      type=str,
    #                      help="The name of the model [default='arrow_tagger']")

    # group.add_option("--label_list",
    #                      dest="label_list",
    #                      default='',
    #                      type=str,
    #                      help="The types of labels to use [default='']")


    config.add_option_group(group)        

if __name__ == '__main__':
    config = initialize_config(argv,params)
    print(config)
    # print(config)
	#main()
