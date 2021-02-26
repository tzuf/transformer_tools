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
from transformers import pipeline

## for the bert prediction
try:
    import spacy
    nlp = spacy.load('en')
except Exception:
    spacy = None

CACHE_SIZE = 10000

@st.cache(allow_output_mutation=True)
def get_cache():
    cache = LRUCache(CACHE_SIZE)
    return cache

def params(config):
    """Main parameters for running the T5 model

    :param config: the global configuration object
    """
    tparams(config)

    group = OptionGroup(config,"transformer_tools.Tagger",
                            "Settings for tagger models")

    config.add_option_group(group)

@st.cache
def mask_prediction_model():
    return pipeline("fill-mask", model="bert-base-uncased")

## caching here causes serious problems 
#@st.cache(allow_output_mutation=True)
def build_config():
    config = initialize_config(sys.argv[1:],params)
    return config

@st.cache(allow_output_mutation=True)
def example_sets():
    return [
        "...",
        "I do not drink coffee at all",
        "Only 12 aliens played with some toys",
        "Tom does not have a sister",
        "I like all vegetables except green cabbage",
        "No alien died without reading some news magazines",
        "At most twelve aliens played with a rock.",
        "At least twelve aliens played with a rock.",
    ]

@st.cache(allow_output_mutation=True)
def build_model(config):
    model = TaggerModel(config)

    ## strange things with streamlit and multiprocessing
    ## see: https://discuss.streamlit.io/t/streamlit-is-stopping/3881/3
    model.model.args.use_multiprocessing=False
    model.model.args.use_multiprocessing_for_evaluation=False
    model.model.args.silent=True

    os.environ["WANDB_MODE"] = "dryrun"
    return (model,model.config.model_name,config.train_name)

def run_model(text_input,mode_set):
    config = build_config()
    model,_,_ = build_model(config)
    cache = get_cache()

    token_tags = cache[text_input]
    if token_tags is None:     
        token_tags = model.query(text_input,convert_to_string=False)
        cache[text_input] = token_tags

    df = pd.DataFrame(
            token_tags,
            columns=["input","tag","probability"],
            index=[str(n) for n in range(len(token_tags))] 
    )

    df2 = None

    if "bert mask prediction" in mode_set:

        mmodel = mask_prediction_model()
        token_list = nlp(text_input)
        #table_substitutions = []
        table_substitutions = cache[token_list]

        if table_substitutions is None:
            table_substitutions = []
            token_indexes = []
            for k,token in enumerate(token_list):
                pos = token.pos_
                if pos == "NOUN" or pos == "VERB" or pos == "ADJ":
                    model_input = f"%s {mmodel.tokenizer.mask_token} %s" %\
                      (' '.join([w.text for w in token_list[:k]]),' '.join([w.text for w in token_list[k+1:]]))
                    substitutions = mmodel(model_input)
                    new_words = [s["token_str"] for s in substitutions]
                    table_substitutions.append((token.text,','.join(new_words)))
                    token_indexes.append(str(k))

            ## add to cache 
            cache[token_list] = table_substitutions

        df2 = pd.DataFrame(
            table_substitutions,
            columns=["token","substitution list"],
            index=token_indexes,
        )
                
                
    return (df,df2)

def main():
    """The Main execution point 
    """
    config = build_config()
    _,mname,tname = build_model(config)
    ex_stories = example_sets()

    st.title("Polarity Projection Interface")

    story_filler = ""
    ex_s = st.selectbox("Select example:",ex_stories,index=0)
    if ex_s != '...':
        story_filler = ex_s
        
    story_text = st.text_area(
        "Sentence Input",story_filler,
        height=50
    )
    
    modes = st.multiselect(
        "Modes",
        ["polarity tagging","bert mask prediction"],
        ["polarity tagging"]
    )
    submit = st.button("Process")

    if story_text and submit:
        mode_set = set(modes)
        with st.spinner("Processing..."):
            df,df2 = run_model(story_text,mode_set)
            st.text('model=%s' % mname)
            st.dataframe(df)

            if df2 is not None:
                st.text('\n\n substitution model=bert')
                st.dataframe(df2)
                
            

if __name__ == "__main__":
    main() 
