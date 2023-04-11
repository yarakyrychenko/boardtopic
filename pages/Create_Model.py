import streamlit as st
import pandas as pd, numpy as np
from bertopic import BERTopic
from transformers import pipeline

def make_stopwords():
    text_file = open("dicts/stopwords.txt", "r")
    stopwords_list = text_file.read().split("\n")
    text_file.close()
    return stopwords_list
stopwords = make_stopwords()

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
    page_title="Create BERTopic",
    page_icon="🤖",
    layout="wide"
)

st.header("🤖 Create BERTopic")
st.subheader("Use this page to create a model with your data")

model_name = st.text_input("Please enter a name for the new model (e.g., 'ukraine_war_jan5')")
df_name =  st.text_input("Please enter data file path (e.g., 'data/df.csv')")
language = st.radio("Please pick one language that best describes your data", ["English","Russian/Ukrainian","Other"],horizontal=True)
datetime_format = st.text_input("Please enter the date format (e.g., '%d.%m.%Y')", value="")
st.session_state.datetime_format = None if datetime_format == "" else datetime_format
embs_name = st.text_input("Please enter embedding file path if any (e.g., 'data/embs.csv')")

if st.button("Train new model"):
    
    from sentence_transformers import SentenceTransformer
    # https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import IncrementalPCA
    from bertopic.vectorizers import OnlineCountVectorizer
    np.random.seed(123)
    from river import cluster
    from helper import River

    umap_model = IncrementalPCA(n_components=5)
    cluster_model = River(cluster.DBSTREAM(clustering_threshold = 1.5,
                                fading_factor = 0.05,
                                cleanup_interval = 7,
                                intersection_factor = 0.5,
                                minimum_weight = 1))
    vectorizer_model = OnlineCountVectorizer(decay=.01,stop_words=stopwords)
    embedding_model = "all-MiniLM-L6-v2" if language=="English" else "paraphrase-multilingual-MiniLM-L12-v2" 
    sentence_model = SentenceTransformer(embedding_model)
    topic_model = BERTopic(verbose=True,
                        embedding_model=embedding_model,
                        umap_model=umap_model,
                        hdbscan_model=cluster_model,
                        vectorizer_model=vectorizer_model,
                        calculate_probabilities=True)

    with st.spinner("Preprocessing..."):
        df = pd.read_csv(df_name)
        new_df = preproc(df)
        new_df['id'] = df.index
        all_docs = list(new_df.proc)
    
    with st.spinner("Generating embeddings. This may take a couple of hours..."):
        try:
            embeddings = np.array(pd.read_csv(embs_name).drop("Unnamed: 0",axis=1))
        except:
            embeddings = sentence_model.encode(new_df.proc, show_progress_bar=True)
            pd.DataFrame(embeddings).to_csv(f"embs_{model_name}.csv")

    with st.spinner("Creating the model. This may take a couple of minutes..."):
        doc_emb_chunks = [(all_docs[i:i+3000],embeddings[i:i+1000]) for i in range(0, len(all_docs), 3000)]
        topics = []
        for doc_chunk, emb_chunk in doc_emb_chunks:
            topic_model.partial_fit(all_docs,embeddings)
            topics.extend(topic_model.topics_)
        topic_model.topics_ = topics
        
    with st.spinner("Classifiying emotions. This may take a couple of minutes..."):
        ems = get_emotions(new_df)
        new_df = pd.merge(new_df, ems, on='id')

    st.session_state.model = topic_model
    st.session_state.df = new_df
    st.session_state.model_df = st.session_state.model.get_document_info(new_df.proc)

    topic_model.save(f"models/{model_name}")
    st.session_state.df.to_csv(f"models/df_{model_name}.csv")
    st.success(f"New model trained and saved as '{model_name}', dataframe saved as 'df_{model_name}.csv' in folter 'models'.")