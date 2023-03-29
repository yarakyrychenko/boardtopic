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

df = pd.read_csv(st.secrets["df"])
model_load = gdown.download(st.secrets["model"], quiet=False)
model = BERTopic(model_load)

topics_over_time = model.topics_over_time(df.original_text, df.date, nr_bins=60,  datetime_format="%d.%m.%Y")

model.visualize_topics_over_time(topics_over_time, top_n_topics=20)

