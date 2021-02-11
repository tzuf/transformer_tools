import streamlit as st 
import os
import sys
import pandas as pd
sys.path.append('.')
from optparse import OptionParser,OptionGroup
from transformer_tools.Tagger import params as tparams
from transformer_tools.Tagger import TaggerModel
from transformer_tools import initialize_config
from transformer_tools.util.cache import LRUCache

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
    return config

@st.cache
def build_model(config):
    model = TaggerModel(config)
    os.environ["WANDB_MODE"] = "dryrun"
    return model

def run_model(text_input,mode_set):
    config = build_config()
    model = build_model(config)
    token_tags = model.query(text_input,convert_to_string=False)

def main():
    """The Main execution point 
    """
    st.title("Polarity Projection Interface")
    story_text = st.text_area(
        "Sentence Input","",
        height=50
    )

    modes = st.multiselect(
        "Modes",
        ["polarity tagging"],
        ["polarity tagging"]
    )
    submit = st.button("Process")

    if story_text and submit:
        mode_set = set(modes)
        with st.spinner("Processing..."):
            run_model(story_text,mode_set)


if __name__ == "__main__":
    main() 
