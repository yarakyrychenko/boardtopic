from river import stream
from river import cluster

class River:
    def __init__(self, model):
        self.model = model

    def partial_fit(self, umap_embeddings):
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            self.model = self.model.learn_one(umap_embedding)

        labels = []
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            label = self.model.predict_one(umap_embedding)
            labels.append(label)

        self.labels_ = labels
        return self


import pandas as pd
from typing import List
import plotly.graph_objects as go
from sklearn.preprocessing import normalize


def visualize_topics_over_time(topic_model,
                               topics_over_time: pd.DataFrame,
                               top_n_topics: int = None,
                               topics: List[int] = None,
                               normalize_frequency: bool = False,
                               custom_labels: bool = False,
                               title: str = "<b>Topics over Time</b>",
                               width: int = 860,
                               height: int = 600) -> go.Figure: 
    """
    Based on BERTopic's funciton https://github.com/MaartenGr/BERTopic/blob/809414b88ca3f12a46728069d098d82345986489/bertopic/plotting/_topics_over_time.py
    """
    #colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#0072B2", "#CC79A7"]

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        selected_topics = list(topics)
    elif top_n_topics is not None:
        selected_topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        selected_topics = sorted(freq_df.Topic.to_list())

    # Prepare data
    if topic_model.custom_labels_ is not None and custom_labels:
        topic_names = {key: topic_model.custom_labels_[key + topic_model._outliers] for key, _ in topic_model.topic_labels_.items()}
    else:
        topic_names = {key: value[:30] + "..." if len(value) > 30 else value
                       for key, value in topic_model.topic_labels_.items()}
    topics_over_time["Name"] = topics_over_time.Topic.map(topic_names)
    data = topics_over_time.loc[topics_over_time.Topic.isin(selected_topics), :].sort_values(["Topic", "Timestamp"])

    # Add traces
    fig = go.Figure()
    for index, topic in enumerate(data.Topic.unique()):
        trace_data = data.loc[data.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            y = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            y = trace_data.Frequency
        fig.add_trace(go.Scatter(x=pd.to_datetime(trace_data.Timestamp), y=y,
                                 mode='lines',
                                 #marker_color=colors[index % 7],
                                 hoverinfo="text",
                                 name=topic_name,
                                 hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words]))

    # Styling of the visualization
    #fig.update_xaxes(
    #    dtick=7,
    #    tickformat="%b\n%Y"
    #     )
    fig.update_layout(
        yaxis_title="Normalized Frequency" if normalize_frequency else "Frequency",
        title={'text':f'{title}',
               'font': dict(size=22)
               },
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            #font_family="Rockwell"
        ),
        legend=dict(
            title="<b>Global Topic Representation",
            orientation="h",
            y = -.2,
            x = 0
            #yanchor="bottom",
            #xanchor="left"
        )
    )
    return fig



def visualize_topics_per_class(topic_model,
                               topics_per_class: pd.DataFrame,
                               top_n_topics: int = 10,
                               topics: List[int] = None,
                               normalize_frequency: bool = False,
                               custom_labels: bool = False,
                               title: str = "<b>Topics per Class</b>",
                               width: int = 900,
                               height: int = 900) -> go.Figure:
    """
    Based on BERTopic's funciton https://github.com/MaartenGr/BERTopic/blob/809414b88ca3f12a46728069d098d82345986489/bertopic/plotting/_topics_per_class.py
    """

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        selected_topics = list(topics)
    elif top_n_topics is not None:
        #selected_topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
        selected_topics = freq_df.Topic.to_list()[:top_n_topics]
    else:
        selected_topics = sorted(freq_df.Topic.to_list())

    # Prepare data
    if topic_model.custom_labels_ is not None and custom_labels:
        topic_names = {key: topic_model.custom_labels_[key + topic_model._outliers] for key, _ in topic_model.topic_labels_.items()}
    else:
        topic_names = {key: value[:40] + "..." if len(value) > 40 else value
                       for key, value in topic_model.topic_labels_.items()}
    topics_per_class["Name"] = topics_per_class.Topic.map(topic_names)
    data = topics_per_class.loc[topics_per_class.Topic.isin(selected_topics), :]

    # Add traces
    fig = go.Figure()
    for index, topic in enumerate(selected_topics):
        if index == 0:
            visible = True
        else:
            visible = "legendonly"
        trace_data = data.loc[data.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            x = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            x = trace_data.Frequency
        fig.add_trace(go.Bar(y=trace_data.Class,
                             x=x,
                             visible=visible,
                             hoverinfo="text",
                             name=topic_name,
                             orientation="h",
                             hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words]))

    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        xaxis_title="Normalized Frequency" if normalize_frequency else "Frequency",
        yaxis_title="Class",
        title={
            'text': f"{title}",
            'font': dict(
                size=22)
        },
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
        ),
        legend=dict(
            title="<b>Global Topic Representation",
        )
    )
    return fig