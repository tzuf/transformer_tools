# Core Pkgs
import streamlit as st 
import os

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

if __name__ == '__main__':
	main()
