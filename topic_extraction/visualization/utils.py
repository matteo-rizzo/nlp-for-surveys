import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP


def visualize_topics_assist(topic_model, umap,
                            topics: list[int] = None,
                            top_n_topics: int = None,
                            custom_labels: bool = False) -> pd.DataFrame:
    """ Visualize topics, their sizes, and their corresponding words

    This visualization is highly inspired by LDAvis, a great visualization
    technique typically reserved for LDA.

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize
        top_n_topics: Only select the top n most frequent topics
        custom_labels: Whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Examples:

    To visualize the topics simply run:

    ```python
    topic_model.visualize_topics()
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_topics()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/viz.html"
    style="width:1000px; height: 680px; border: 0px;""></iframe>
    """
    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Extract topic words and their frequencies
    topic_list = sorted(topics)
    frequencies = [topic_model.topic_sizes_[topic] for topic in topic_list]
    if custom_labels and topic_model.custom_labels_ is not None:
        words = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in topic_list]
    else:
        words = [" | ".join([word[0] for word in topic_model.get_topic(topic)[:5]]) for topic in topic_list]

    # Embed c-TF-IDF into 2D
    all_topics = sorted(list(topic_model.get_topics().keys()))
    indices = np.array([all_topics.index(topic) for topic in topics])
    embeddings = topic_model.c_tf_idf_.toarray()[indices]
    embeddings = MinMaxScaler().fit_transform(embeddings)
    print(embeddings.shape)
    # if not umap.fitted:
    #     umap.fit(embeddings)
    #     umap.fitted = True
    embeddings = umap.fit_transform(embeddings)

    # Visualize with plotly
    df = pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1],
                       "Topic": topic_list, "Words": words, "Size": frequencies})
    return df
