# Core Pkgs
import streamlit as st
import os
import sys
import pandas as pd
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('.')
from optparse import OptionParser, OptionGroup
from transformer_tools.T5Classification import params as tparams
from transformer_tools import initialize_config
from transformer_tools import LoadT5Classifier
from transformer_tools.util.cache import LRUCache

BOILER_PLATE = {
    "Enter your story": "",
    "Enter question (optional)": "",
}

CACHE_SIZE = 10000


def params(config):
    """Main parameters for running the T5 model
    :param config: the global configuration object
    """
    tparams(config)

    group = OptionGroup(config, "transformer_tools.Tagger",
                        "Settings for tagger models")

    config.add_option_group(group)


@st.cache(allow_output_mutation=True)
def build_config():
    config = initialize_config(sys.argv[1:], params)
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
    return (
        [
            "...",
            "Julie and John moved to the school. Afterwards they journeyed to the park. Fred and Daniel  moved to the bedroom. Mary went back to the office. John went to the kitchen. Following that he went back to the hallway. Mary got the football. Mary travelled to the garden. Following that she moved to the bedroom. Fred went to the school. Fred grabbed the milk. Fred discarded the milk. Bill is either in the bedroom or the school. Mary gave the football to Daniel. Daniel dropped the football. Fred is in the office. Sandra journeyed to the cinema. Julie and Daniel went back to the bathroom. Daniel and John travelled to the office. Afterwards they moved to the park. Julie is either in the office or the school. Bill and Mary travelled to the garden. Then they journeyed to the hallway. Mary and Bill journeyed to the school. Sandra and Mary moved to the park. Then they went to the school. Sandra journeyed to the garden. After that she went to the bathroom. Sandra went to the garden. After that she moved to the bathroom.",
            "Sandra took the milk. Mary and Daniel went to the cinema. After that they moved to the office. Mary is either in the kitchen or the cinema. Julie and Daniel went to the kitchen. Following that they moved to the cinema. Jeff and Julie journeyed to the office. Afterwards they moved to the hallway. Mary is not in the cinema. Fred and Julie went back to the cinema. Sandra put down the milk. Following that she travelled to the hallway. Mary is either in the school or the kitchen. John is either in the school or the office. Julie is in the park. Julie is either in the cinema or the kitchen. John and Sandra went back to the cinema. Afterwards they travelled to the park. Julie and Fred went back to the hallway. Sandra went back to the school. After that she went back to the park. Jeff and Mary moved to the park. Jeff went to the school. Bill and Daniel journeyed to the bathroom. Following that they went back to the office. Daniel took the football. Afterwards he discarded the football. Fred journeyed to the kitchen. Then he went to the cinema. Fred and Bill went to the kitchen.",
            "Mark is in the garden. John traveled to the store. He then returned home.",
        ],
        [
            "...",
            "Where is the milk?",
            "Is Fred in the kitchen?",
            "Where is John?",
        ])


def run_model(mode_set,
              story_text,
              question,
              layer):
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
    row = [];
    index = []
    row_der = [];
    index_der = []

    ## story + question => answer
    q_input = "%s $question$ %s" % (story_text, question)
    q_attention = "%s $attention$ %s" % (story_text, question)
    q_tokens = "%s $tokens$ %s" % (story_text, question)

    answer_text = cache[q_input]
    cross_attentions = cache[q_attention]
    input_tokens = cache[q_tokens]

    if answer_text is None or cross_attentions is None or input_tokens is None:
        answer_text, cross_attentions, input_tokens, _ = model.query(q_input)
        # answer_text = model.query(q_input)[0]
        cache[q_input] = answer_text
        cache[q_attention] = cross_attentions
        cache[q_tokens] = input_tokens


    if "s+q => a" in mode_set:
        row.append([q_input, answer_text])
        index.append("s+q => a")
        # cache[q_input] = answer_text

    if "s => ql" in mode_set:
        inquire_prompt = "$story$ %s" % (story_text)
        inquire_text = cache[inquire_prompt]
        if inquire_text is None:
            inquire_text = model.query(inquire_prompt, prefix="inquire:")[0]
            cache[inquire_prompt] = inquire_text

        row.append([inquire_prompt, inquire_text])
        index.append("s => ql")

    if "s+p => g" in mode_set:
        location_prompt = "$prompt$ %s $story$ %s" % (question, story_text)
        location_text = cache[location_prompt]
        if location_text is None:
            location_text = model.query(location_prompt, prefix="locate:")[0]
            cache[location_prompt] = location_text
        row.append([location_prompt, location_text])
        index.append("s+p => g")

    ## other modes that rely on supporting facts
    if "s+q => sp" in mode_set or \
            "sp+s+q => a" in mode_set or \
            "s+a+sp => q" in mode_set:
        supporting_text = "$question$ %s $story$ %s" % (question, story_text)
        support_markedup = "$question$ %s $story$ %s" % (question, story_text)
        support_out = cache[supporting_text]
        if support_out is None:
            support_out = model.query(supporting_text, prefix="generate:")[0]
            cache[supporting_text] = support_out

        if "s+q => sp" in mode_set:
            row.append([supporting_text, support_out])
            index.append("s+q => sp")

        ##
        if "sp+s+q => a" in mode_set:
            full_input = "$context$ %s $story$ %s $question$ %s" % (support_out, story_text, question)
            second_answer_out = cache[full_input]
            if second_answer_out is None:
                second_answer_out = model.query(full_input, prefix="answer:")[0]
                cache[full_input] = second_answer_out

            row_der.append([full_input, second_answer_out])
            index_der.append("sp+s+q => a")
        if "s+a+sp => q" in mode_set:
            question_input = "$answer$ %s $context$ %s $story$ %s" % (answer_text, support_out, story_text)
            question_out = cache[question_input]
            if question_out is None:
                question_out = model.query(question_input, prefix="question:")[0]
                cache[question_input] = question_out

            row_der.append([question_input, question_out])
            index_der.append("s+a+sp => q")

    heads = list(map(str, list(range(0,12))))

    answer_tokens = ['▁'+ans for ans in answer_text]
    fig = plot(cross_attentions.cpu().squeeze(), input_tokens[0], heads, layer=layer, answers=answer_tokens)

    df = pd.DataFrame(
        row,
        columns=["input", "output (predicted)"],
        index=index
    )

    df2 = None
    if index_der and row_der:
        df2 = pd.DataFrame(
            row_der,
            columns=["input (predicted)", "output (predicted)"],
            index=index_der
        )
    return (df, df2, fig)


def main():
    # config = build_config()
    # model = build_model(config)
    ex_stories, ex_questions = example_sets()
    st.title("T5 bAbi Interface")

    ## page details
    story_filler = "Enter your story"
    q_filler = "Enter your question"
    ex_s = st.selectbox("Select example:", ex_stories, index=0)
    if ex_s != '...':
        story_filler = ex_s
        q_filler = ex_questions[ex_stories.index(ex_s)]

    story_text = st.text_area(
        "story text", story_filler,
        height=400
    )
    story_text = BOILER_PLATE.get(story_text, story_text)

    question = st.text_area(
        "question text",
        q_filler,
        height=1
    )
    question = BOILER_PLATE.get(question, question)


    layer = st.selectbox(
        "Layer number (0-11)",
        list(range(0, 12)),
        index=10

    )


    # story + answer + supporting facts => question
    modes = st.multiselect(
        "Modular Computations (story=`s`, question=`q`, supporting sentences=`sp`, answer=`a`, question list=`ql`, prompt=`p`, graph=`g`)",
        ["s+q => a", "s+q => sp", "sp+s+q => a", "s+a+sp => q", "s => ql", "s+p => g"],
        ["s+q => a"]
    )

    ## answer a query
    submit = st.button("Process")
    row = [];
    index = []
    row_der = [];
    index_der = []

    if story_text and question and submit:
        mode_set = set(modes)
        with st.spinner("Processing..."):
            df, df2, fig = run_model(mode_set, story_text, question, layer)


            # main computations
            st.write("<b> Direct Computations </b>", unsafe_allow_html=True)
            st.table(df)
            # st.dataframe(df,width=400)

            ## derived
            if df2 is not None:
                st.write("<b> Derived Computations/Round-trips </b>", unsafe_allow_html=True)
                st.table(df2)

            if story_text not in ex_stories:
                ex_stories.append(story_text)
                ex_questions.append(question)
            st.pyplot(fig)

            # st.success("Finished!")


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, shrink=1)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #          rotation_mode="anchor")
    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), fontsize=12, **kw)
            texts.append(text)

    return texts

def plot(attention, tokens, heads, layer, answers):
    attention = attention[0, layer, :, :]

    attention = np.transpose(np.round(attention, 2), (1, 0))
    fig, ax = plt.subplots()

    im = heatmap(attention, tokens, heads, ax=ax, cmap="YlGn")

    texts = annotate_heatmap(im, valfmt="{x:.1f}")

    len_tokens = len(tokens)
    fig.tight_layout(pad=-2*len_tokens)
    # fig.set_size_inches(20, 30)

    plt.xlabel("Attention heads")
    plt.ylabel("Input")

    plt.title(f"Attention between Output (decoded) and Input (encoded) over layer {layer}", pad=10, fontsize=14)

    # plt.show()
    # Set the gold lable - blue

    indices = [index for index, element in enumerate(tokens) for answer in answers if element == answer]

    for idx in indices:
        t = ax.yaxis.get_ticklabels()[idx]
        t.set_color('blue')
        t.set_fontweight('bold')


    plt.savefig('saved_figure.png')
    return fig


if __name__ == '__main__':
    main()