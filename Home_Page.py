import streamlit as st
import pandas as pd, numpy as np
from bertopic import BERTopic
from datetime import datetime
import math
from helper import visualize_topics_over_time, visualize_topics_per_class

@st.cache_data
def get_df(url):
    df = pd.read_csv(url)
    return df
    
@st.cache_resource
def get_model(url):
    return BERTopic.load(url)

@st.cache_data
def get_topics_over_time(frame,lens):
    strings = frame.proc2.apply(lambda x: str(x))
    date = pd.to_datetime(frame.date)
    return st.session_state.model.topics_over_time(strings, date, nr_bins=math.floor(len(frame.date.unique())/3))

@st.cache_data
def get_topics_per_class(frame,colname):
    strings = frame.proc2.apply(lambda x: str(x))
    classes = st.session_state.df[colname].apply(lambda x: str(x))
    return st.session_state.model.topics_per_class(strings, classes=classes)

st.set_page_config(
    page_title="BoardTopic",
    page_icon="🤖",
   layout="wide"
)

st.header("🤖 BoardTopic")
st.subheader("Turning your data into insight with behavioral data science")

if "model" not in st.session_state:
    st.markdown("Welcome to BoardTopic, a friendly way to understand your big data.")
    st.markdown("If you do not have a BoardTopic model trained, please go to the 'Create Model' tab.")
    st.markdown("If you already have a BoardTopic model trained, please enter the information below:")
    model_name = st.text_input("Please enter model file name (e.g., 'model')")
    df_name =  st.text_input("Please enter dataframe file name (e.g., 'df_small.csv')")
    if st.button("Enter"):
        st.session_state.model = get_model(f'models/{model_name}')
        st.session_state.df = get_df(f'models/{df_name}')
        st.success("Model and dataframe loaded!")
if "model" in st.session_state:
    #st.session_state.df = get_df("df_small.csv")
    st.session_state.model.set_topic_labels(st.session_state.model.generate_topic_labels(nr_words=6, topic_prefix=False, word_length=10, separator=", "))
    st.session_state.model_df = st.session_state.model.get_document_info(st.session_state.df.proc)
    st.session_state.df["id"] = st.session_state.model_df.index
    st.session_state.model_df["id"] = st.session_state.model_df.index
    st.session_state.model_df = pd.merge(st.session_state.model_df,st.session_state.df,how="left",on="id")
    st.session_state.model_df["date"] =  pd.to_datetime(st.session_state.model_df.date)

    topics_over_time = get_topics_over_time(st.session_state.df,len(st.session_state.df))
    largest_topics  = st.session_state.model_df.groupby("Topic").agg("count").sort_values("Document",ascending=False)[0:10]

    st.write(visualize_topics_over_time(st.session_state.model, topics_over_time, topics=list(largest_topics.index),
                                                           custom_labels=True, title = "10 most popular narratives over time"))

    st.markdown("#### Overall document distribution")

    grouped = st.session_state.model_df.groupby("date").agg("count")
    grouped['date'] = pd.to_datetime(grouped.index)
    st.bar_chart(data=grouped, x='date', y='Document')

    st.markdown("#### Emotions")

    joy = st.session_state.model_df.joy.apply(lambda x: 1 if x > 0.9 else 0)
    sadness = st.session_state.model_df.sadness.apply(lambda x: 1 if x > 0.9 else 0)
    surprise = st.session_state.model_df.surprise.apply(lambda x: 1 if x > 0.9 else 0)
    fear = st.session_state.model_df.fear.apply(lambda x: 1 if x > 0.9 else 0)
    anger = st.session_state.model_df.anger.apply(lambda x: 1 if x > 0.9 else 0)

    emotions = pd.DataFrame({"date":st.session_state.model_df.date, "source": st.session_state.model_df.source,
                        "joy":joy, "sadness":sadness, "surprise":surprise, "fear":fear, "anger":anger})
#dates = pd.to_datetime(emotions.date.unique(),format="%d.%m.%Y").sort_values()
#emotions["date"] = pd.to_datetime(emotions.date,format="%d.%m.%Y")
#emnew = emotions[(dates[-7] <= emotions.date) & (emotions.date <= dates[-1])].drop('date',axis=1, inplace=False).mean()
#emplot = pd.DataFrame({f"Week of {str(dates[-14])[:10]}": emold, f"Week of {str(dates[-7])[:10]}": emnew}).T

    st.markdown("##### Percent with emotion by platform")
    st.bar_chart(emotions.groupby("source").agg("mean").T*100)

    st.markdown("##### Platform breakdown")
    st.bar_chart(emotions.groupby("source").agg("mean")*100)

    emotionsgr = emotions.groupby("date").agg("mean")*100
    emotionsgr['date'] = pd.to_datetime(grouped.index)

    st.markdown("##### Emotional dynamics over time")
    st.line_chart(emotionsgr,x="date")

    st.markdown("#### Topics per class")
    if "source" in st.session_state.df.columns:
        topics_per_class1 = get_topics_per_class(st.session_state.df,"source")
        st.plotly_chart(visualize_topics_per_class(st.session_state.model, topics_per_class1, top_n_topics=20, width = 900, height = 600,
                                               custom_labels=True, title = "20 most popular narratives per platform"))
    st.session_state.df["emotion"] = st.session_state.df[["joy","sadness","surprise","fear",'anger','no_emotion']].idxmax(axis=1)
    topics_per_class2 = get_topics_per_class(st.session_state.df,"emotion")
    st.plotly_chart(visualize_topics_per_class(st.session_state.model, topics_per_class2, top_n_topics=20, width = 900, height = 600,
                                               custom_labels=True, title = "20 most popular narratives per emotion"))

    st.markdown("#### All topics")
    last_week = st.session_state.model_df
    largest_topics_last_week = last_week.groupby("Topic").agg("count").sort_values("Document",ascending=False)
    largest_topics_last_week["Name"] = [ list(last_week[last_week.Topic == i]["CustomName"])[0] for i in largest_topics_last_week.index ] 
    largest_topics_last_week["Count"] = largest_topics_last_week["Document"] 
    largest_topics_last_week["Percent"] = round(100*largest_topics_last_week["Count"]/len(st.session_state.model_df),3)
    st.table(largest_topics_last_week[["Name", "Count","Percent"]])

    dictionary = {i:st.session_state.model.custom_labels_[i] for i in range(len(st.session_state.model.custom_labels_))}
    def mapping(item):
        return dictionary[item]

    st.markdown("#### Explore representative documents")
    st.selectbox("Select topic",list(st.session_state.model_df.Topic.unique()),key="selected_topic",format_func=mapping)
    repr_docs_mappings, repr_docs, repr_docs_indices = st.session_state.model._extract_representative_docs(st.session_state.model.c_tf_idf_,st.session_state.model_df,st.session_state.model.topic_representations_)
    ind = repr_docs_indices[st.session_state.selected_topic]
    j = 1
    for doc in st.session_state.model_df.iloc[ind].Document:
        st.markdown(f"**Representative document {j}**")
        st.text(doc)
        j+=1

    st.markdown("---")
    st.markdown("### Save current model")
    name = st.text_input("Please name this model file (e.g., 'my_cool_model')")
    if st.button("Save this model"):
        st.session_state.model.save(f"models/model_{name}")
        st.session_state.df.to_csv(f"models/df_{name}.csv")
        st.success(f"Model and dataframe saved in folder 'models'!")
    if st.button("Restart"):
        st.cache_data.clear()
        st.cache_resource.clear()
        for key in st.session_state.keys():
            del st.session_state[key]
        
