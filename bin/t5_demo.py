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
    config.max_answer = 100
    config.max_seq_len = 300
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
    return ([
    "...",
    "Julie and John moved to the school. Afterwards they journeyed to the park. Fred and Daniel  moved to the bedroom. Mary went back to the office. John went to the kitchen. Following that he went back to the hallway. Mary got the football. Mary travelled to the garden. Following that she moved to the bedroom. Fred went to the school. Fred grabbed the milk. Fred discarded the milk. Bill is either in the bedroom or the school. Mary gave the football to Daniel. Daniel dropped the football. Fred is in the office. Sandra journeyed to the cinema. Julie and Daniel went back to the bathroom. Daniel and John travelled to the office. Afterwards they moved to the park. Julie is either in the office or the school. Bill and Mary travelled to the garden. Then they journeyed to the hallway. Mary and Bill journeyed to the school. Sandra and Mary moved to the park. Then they went to the school. Sandra journeyed to the garden. After that she went to the bathroom. Sandra went to the garden. After that she moved to the bathroom.",
    "Sandra took the milk. Mary and Daniel went to the cinema. After that they moved to the office. Mary is either in the kitchen or the cinema. Julie and Daniel went to the kitchen. Following that they moved to the cinema. Jeff and Julie journeyed to the office. Afterwards they moved to the hallway. Mary is not in the cinema. Fred and Julie went back to the cinema. Sandra put down the milk. Following that she travelled to the hallway. Mary is either in the school or the kitchen. John is either in the school or the office. Julie is in the park. Julie is either in the cinema or the kitchen. John and Sandra went back to the cinema. Afterwards they travelled to the park. Julie and Fred went back to the hallway. Sandra went back to the school. After that she went back to the park. Jeff and Mary moved to the park. Jeff went to the school. Bill and Daniel journeyed to the bathroom. Following that they went back to the office. Daniel took the football. Afterwards he discarded the football. Fred journeyed to the kitchen. Then he went to the cinema. Fred and Bill went to the kitchen.",
    "Mark is in the garden. John traveled to the store. He then returned home.",
    ],[
        "...",
        "Where is the milk?",
        "Is Fred in the kitchen?",
        "Where is John?",
    ])

def run_model(mode_set,
                  story_text,
                  question):
    """Main method for running the model 

    :param mode_set: the modes to run
    :param story_text: the main story text 
    :param question: the typed question 
    :returns: two data frames
    :rtype: tuple 
    """
    config = build_config()
    model = build_model(config)
    cache = get_cache()

    ## for building data frames
    row = []; index = []
    row_der = []; index_der = []

    ## story + question => answer 
    q_input = "%s $question$ %s" % (story_text,question)
    answer_text = cache[q_input]
    if answer_text is None:
        answer_text = model.query(q_input)[0]
        cache[q_input] = answer_text

    if "s+q => a" in mode_set: 
        row.append([q_input,answer_text])
        index.append("s+q => a")
        #cache[q_input] = answer_text

    if "s => ql" in mode_set:
        inquire_prompt = "$story$ %s" % (story_text)
        inquire_text = cache[inquire_prompt]
        if inquire_text is None: 
            inquire_text = model.query(inquire_prompt,prefix="inquire:")[0]
            cache[inquire_prompt] = inquire_text
        
        row.append([inquire_prompt,inquire_text])
        index.append("s => ql")

    if "s+p => g" in mode_set:
        location_prompt = "$prompt$ %s $story$ %s" % (question,story_text)
        location_text = cache[location_prompt]
        if location_text is None:
            location_text = model.query(location_prompt,prefix="locate:")[0]
            cache[location_prompt] = location_text
        row.append([location_prompt,location_text])
        index.append("s+p => g")

    ## other modes that rely on supporting facts 
    if "s+q => sp" in mode_set or \
      "sp+s+q => a" in mode_set or \
      "s+a+sp => q" in mode_set:
        supporting_text = "$question$ %s $story$ %s" % (question,story_text)
        support_markedup = "$question$ %s $story$ %s" % (question,story_text)
        support_out = cache[supporting_text]
        if support_out is None:
            support_out = model.query(supporting_text,prefix="generate:")[0]
            cache[supporting_text] = support_out
                
        if "s+q => sp" in mode_set:
            row.append([supporting_text,support_out])
            index.append("s+q => sp")

        ##
        if "sp+s+q => a" in mode_set:
            full_input = "$context$ %s $story$ %s $question$ %s" % (support_out,story_text,question)
            second_answer_out = cache[full_input]
            if second_answer_out is None:
                second_answer_out = model.query(full_input,prefix="answer:")[0]
                cache[full_input] = second_answer_out

            row_der.append([full_input,second_answer_out])
            index_der.append("sp+s+q => a")
        if "s+a+sp => q" in mode_set:
            question_input = "$answer$ %s $context$ %s $story$ %s" % (answer_text,support_out,story_text)
            question_out = cache[question_input]
            if question_out is None: 
                question_out = model.query(question_input,prefix="question:")[0]
                cache[question_input] = question_out
                
            row_der.append([question_input,question_out])
            index_der.append("s+a+sp => q")

    df = pd.DataFrame(
            row,
            columns=["input","output (predicted)"],
            index=index
        )

    df2 = None
    if index_der and row_der:
        df2 = pd.DataFrame(
                row_der,
                columns=["input (predicted)","output (predicted)"],
                index=index_der
        )

    return (df,df2)
            

def main():
    config = build_config()
    model = build_model(config)
    ex_stories,ex_questions = example_sets()
    st.title("T5 bAbi Interface")

    ## page details
    story_filler = "Enter your story"
    q_filler = "Enter your question"
    ex_s = st.selectbox("Select example:",ex_stories,index=0)
    if ex_s != '...':
        story_filler = ex_s
        q_filler = ex_questions[ex_stories.index(ex_s)]
    
    story_text = st.text_area(
        "story text",story_filler,
        height=400
    )
    story_text = BOILER_PLATE.get(story_text,story_text)

    question = st.text_area(
        "question text",
        q_filler,
        height=1
    )
    question = BOILER_PLATE.get(question,question)

    #story + answer + supporting facts => question 
    modes = st.multiselect(
        "Modular Computations (story=`s`, question=`q`, supporting sentences=`sp`, answer=`a`, question list=`ql`, prompt=`p`, graph=`g`)",
        ["s+q => a","s+q => sp","sp+s+q => a","s+a+sp => q","s => ql", "s+p => g"],
        ["s+q => a"]
    )

    ## answer a query
    submit = st.button("Process")
    row = []; index = []
    row_der = []; index_der = []

    if story_text and question and submit:
        mode_set = set(modes)
        with st.spinner("Processing..."):
            df,df2 = run_model(mode_set,story_text,question)

            # main computations
            st.write("<b> Direct Computations </b>",unsafe_allow_html=True)
            st.table(df)
            #st.dataframe(df,width=400)

            ## derived
            if df2 is not None:
                st.write("<b> Derived Computations/Round-trips </b>",unsafe_allow_html=True)
                st.table(df2)

            if story_text not in ex_stories:
                ex_stories.append(story_text)
                ex_questions.append(question)

            st.success("Finished!")
        
if __name__ == '__main__':
    main()
