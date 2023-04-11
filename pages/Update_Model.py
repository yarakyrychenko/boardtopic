import streamlit as st
import pandas as pd, numpy as np
from bertopic import BERTopic
from transformers import pipeline

@st.cache_data
def get_emotions(frame, language):
    clasif = "cointegrated/rubert-tiny2-cedr-emotion-detection" if language == "Russian/Ukrainian" else "j-hartmann/emotion-english-distilroberta-base"
    st.classifier = pipeline("text-classification", model=clasif, return_all_scores=True)
    temp = st.classifier(list(frame.proc2))
    rangelabels = len(temp[0])
    temp = pd.DataFrame({temp[0][j]["label"]: [ temp[i][j]["score"] for i in range(len(temp)) ] for j in range(rangelabels)})
    temp['id'] = [i for i in range(len(st.session_state.df),len(temp)+len(st.session_state.df))]
    return temp

def preproc(frame):
    import re
    frame["proc"] = frame.text.apply(lambda x: str(x))
    frame.proc = frame.apply(lambda row: re.sub(r"http\S+", "http", row.proc), 1)
    frame.proc = frame.apply(lambda row: re.sub(r"@\S+", "@user", row.proc), 1)
    frame.proc = frame.apply(lambda row: re.sub(r"#", " ", row.proc).strip(), 1)
    frame["proc2"] = frame.proc
    frame.proc2 = frame.proc2.apply(lambda row: row[:2048].lower())
    return frame

st.set_page_config(
    page_title="Update BERTopic",
    page_icon="🤖",
    layout="wide"
)

st.header("🤖 Update BERTopic")
st.subheader("Use this page to update your model with new data")

if "model" not in st.session_state:
    st.markdown("**No model detected. Please go to the Home Page and add a model first.**")
if "model" in st.session_state:
    old_df = st.session_state.df
    topics = list(st.session_state.model.topics_)
    st.markdown(f"**Current data:** {len(old_df)} rows, {len(st.session_state.model.topic_labels_)} topics.")
    st.markdown(f"**Current date range:** {str(min(pd.to_datetime(old_df.date, format='%d.%m.%Y')))[:10]} -- {str(max(pd.to_datetime(old_df.date,format='%d.%m.%Y')))[:10]}.")
    st.write(old_df)

    st.markdown("#### Please eneter a name for the updated model and upload files below")
    name = st.text_input("Please enter a model name (e.g., 'my_cool_model')")
    language = st.radio("Please pick one language that best describes your data", ["English","Russian/Ukrainian","Other"],horizontal=True)
    datetime_format = st.text_input("Please enter the date format (e.g., '%d.%m.%Y')", value="")
    st.session_state.datetime_format = None if datetime_format == "" else datetime_format
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    if st.button('All files selected'):
        for i in range(len(uploaded_files)):
            uploaded_file = uploaded_files[i]
            new_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded file {uploaded_file.name}")

            with st.spinner("Preprocessing..."):
                new_df= preproc(new_df)
                new_df['id'] = [i for i in range(len(old_df),len(new_df)+len(old_df))]
                docs = list(new_df.proc)

            with st.spinner("Updating the model. This may take a couple of minutes..."):
                st.session_state.model.partial_fit(docs)
                topics.extend(st.session_state.model.topics_)

            with st.spinner("Classifiying emotions. This may take a couple of minutes..."):
                ems = get_emotions(new_df, language)
                new_df = pd.merge(new_df, ems, on='id')

            old_df = pd.concat([old_df,new_df])
            st.success(f"Done with file {uploaded_file.name}!")
            if i == len(uploaded_files)-1:
                st.session_state.df = old_df
                st.session_state.model.topics_ = topics

                st.session_state.model.set_topic_labels(st.session_state.model.generate_topic_labels(nr_words=5, topic_prefix=False, word_length=10, separator=", "))
                st.session_state.model_df = st.session_state.model.get_document_info(st.session_state.df.proc)
                st.session_state.df["id"] = st.session_state.model_df.index
                st.session_state.model_df["id"] = st.session_state.model_df.index
                st.session_state.model_df = pd.merge(st.session_state.model_df,st.session_state.df,how="left",on="id")
                st.session_state.model_df["date"] =  pd.to_datetime(st.session_state.model_df.date, format="%d.%m.%Y") 
                
                st.markdown("---")
                st.markdown(f"**Updated data:** {len(old_df)} rows, {len(st.session_state.model.topic_labels_)} topics.")
                st.markdown(f"**Updated date range:** {str(min(pd.to_datetime(old_df.date, format='%d.%m.%Y')))[:10]} -- {str(max(pd.to_datetime(old_df.date,format='%d.%m.%Y')))[:10]}.")

                st.session_state.model.save(f"models/model_{name}")
                st.session_state.df.to_csv(f"models/df_{name}.csv")
                st.success(f"Model and dataframe saved in folder 'model'!")
            


