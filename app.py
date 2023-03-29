import streamlit as st
import pandas as pd, numpy as np
from bertopic import BERTopic
import gdown


st.set_page_config(
    page_title="BoardTopic",
    page_icon="🤖",
    #layout="wide"
)

st.header("🤖 BoardTopic")
st.subheader("Interactive online topic model and dashboard for BERTopic")

@st.cache_data
def get_df(url):
    return pd.read_csv(url)
    
@st.cache_resource
def get_model(url):
    model_load = gdown.download(url, quiet=False)
    return BERTopic.load(model_load)
    
df = get_df(st.secrets["df"])
model = get_model(st.secrets["model"])
st.markdown("yay")

topics_over_time = model.topics_over_time(df.original_text, df.date, nr_bins=60,  datetime_format="%d.%m.%Y")
st.markdown("yay2")
model.visualize_topics_over_time(topics_over_time, top_n_topics=20)

