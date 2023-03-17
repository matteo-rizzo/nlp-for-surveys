from __future__ import annotations

import datetime
from pathlib import Path
from pprint import pprint

import numpy as np
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from topic_extraction.classes import Document
from topic_extraction.classes.BaseTopicExtractor import BaseTopicExtractor
from topic_extraction.utils import load_yaml
from sklearn.metrics.pairwise import cosine_similarity


class BERTopicExtractor(BaseTopicExtractor):
    def save(self, path: str | Path, *args, **kwargs):
        self._topic_model.save(path, *args, **kwargs)

    def load(self, path: str | Path, *args, **kwargs):
        self._topic_model = BERTopic.load(path, *args, **kwargs)
        # TODO: assign components

    def __init__(self):
        self.__train_embeddings = None
        self._topic_model: BERTopic = None
        self._reduction_model = None
        self._config = None
        self._embedding_model = None
        self._clustering_model = None
        self._vectorizer_model = None
        self._weighting_model = None
        self._representation_model = None
        self._reduce_outliers: bool = False

    def prepare(self, *args, **kwargs):
        config_path: str | Path = kwargs.get("config_file")
        self._config = load_yaml(config_path)

        run_config = self._config["run"]
        model_config = self._config["model"]

        print("*** Preparing everything ***")

        # Step 1 - Extract embeddings
        self._embedding_model = SentenceTransformer(model_config["sentence_transformer"])

        # Step 2 - Reduce dimensionality
        conf = model_config["dimensionality_reduction"]
        model_rd = None
        if conf["choice"] == "umap":
            model_rd = UMAP(**conf["params"][conf["choice"]])
        elif conf["choice"] == "pca":
            model_rd = PCA(**conf["params"][conf["choice"]])
        self._reduction_model = model_rd

        # Step 3 - Cluster reduced embeddings
        conf = model_config["clustering"]
        model_cl = None
        if conf["choice"] == "hdbscan":
            model_cl = HDBSCAN(**conf["params"][conf["choice"]])
        self._clustering_model = model_cl
        # is UMAP.n_components is increased may want to change metric in HDBSCAN

        # Step 4 - Tokenize topics
        self._vectorizer_model = CountVectorizer(**model_config["vectorizer"]["params"])

        # Step 5 - Create topic representation
        self._weighting_model = ClassTfidfTransformer(**model_config["weighting"]["params"])

        # Step 6 - (Optional) Fine-tune topic representations
        conf = model_config["representation"]
        model_ft = None
        if conf["choice"] == "mmr":
            model_ft = MaximalMarginalRelevance(**conf["params"][conf["choice"]])
        elif conf["choice"] == "keybert":
            model_ft = KeyBERTInspired(**conf["params"][conf["choice"]])
        self._representation_model = model_ft

        tl_args = dict(
            embedding_model=self._embedding_model,  # Step 1 - Extract embeddings
            vectorizer_model=self._vectorizer_model,  # Step 4 - Tokenize topics
            ctfidf_model=self._weighting_model,  # Step 5 - Extract topic words
        )
        if self._reduction_model is not None:
            tl_args["umap_model"] = self._reduction_model  # Step 2 - Reduce dimensionality
        if self._clustering_model is not None:
            tl_args["hdbscan_model"] = self._clustering_model  # Step 3 - Cluster reduced embeddings
        if self._representation_model is not None:
            tl_args["representation_model"] = self._representation_model  # Step 6 - (Optional) Fine-tune topic representations

        self._topic_model = BERTopic(
            **tl_args,
            **model_config["bertopic"]
        )

        self._reduce_outliers = run_config["reduce_outliers"]

    def train(self, documents: list[Document], *args, **kwargs) -> None:
        texts = [d.body for d in documents]

        print("*** Generating embeddings ***")

        # Precompute embeddings
        embeddings = self._embedding_model.encode(texts, show_progress_bar=False)
        self.__train_embeddings = embeddings

        print("*** Fitting the model ***")

        # Topic modelling
        # topics, probs = \
        self._topic_model.fit(texts, embeddings=embeddings)

    def extract(self, document: Document, k: int, *args, **kwargs) -> list:
        pass

    def batch_extract(self, documents: list[Document], k: int, *args, **kwargs) -> tuple:
        """
        Compute topic clusters for each document
        
        :param documents: document to label
        :param k: ignored
        :param use_training_embeddings: if true assumes documents are the same used for training, otw must be set to false
        :param args: 
        :param kwargs: 
        :return:
        """
        print("*** Extracting topics ***")

        emb_train: bool = kwargs.get("use_training_embeddings", False)
        texts = [d.body for d in documents]

        emb = self.__train_embeddings if emb_train else None

        topics, probs = self._topic_model.transform(texts, embeddings=emb)

        if self._reduce_outliers:
            print("*** Reducing outliers ***")
            topics = self._topic_model.reduce_outliers(texts, topics)

        return topics, probs, self._topic_model.get_topics()

    def see_topic_evolution(self, documents: list[Document], bins_n: int):

        timestamps: list[int] = [d.timestamp for d in documents]

        # Create bins of timestamps
        bins = np.linspace(min(timestamps), max(timestamps), bins_n)

        # Assign documents to bins
        doc_bins = np.digitize(timestamps, bins)

        # Extract text for each bin
        doc_by_bin: dict[int, list[str]] = {}
        for doc, idx in zip(documents, doc_bins.tolist()):
            docs = doc_by_bin.setdefault(idx, list())
            docs.append(doc.body)

        model_prev = None
        next_topic: list[list[tuple[int, str]]] | None = None
        # Clustering
        for idx, docs in doc_by_bin.items():
            t_model = BERTopic(embedding_model=self._embedding_model, umap_model=self._reduction_model)
            t_model.fit(docs)

            if model_prev is None:
                # initialize topics
                names = t_model.get_topic_info()["Name"].tolist()[1:]
                print(names)
                next_topic = [
                    [(int(k), names[k])] for k in t_model.get_topics().keys()
                ]
            else:
                sim_matrix = cosine_similarity(model_prev.topic_embeddings_, t_model.topic_embeddings_)
                for i in range(len(next_topic)):
                    last_topic_entry: int = next_topic[i][-1][0]
                    most_similar_topic: int = np.argmax(sim_matrix[last_topic_entry + 1]) - 1
                    names = t_model.get_topic_info()["Name"].tolist()[1:]
                    print(names)
                    print(most_similar_topic)
                    next_topic[i].append((most_similar_topic, names[most_similar_topic]))

            model_prev = t_model

        # Print topics to file
        pprint(next_topic)

    def plot_wonders(self, documents: list[Document], **kwargs) -> None:

        print("*** Plotting results ***")

        emb_train: bool = kwargs.get("use_training_embeddings", False)
        path = Path("plots")

        path.mkdir(parents=True, exist_ok=True)

        formatted_labels = self._topic_model.generate_topic_labels(nr_words=4, topic_prefix=False, word_length=16, separator=" - ")
        self._topic_model.set_topic_labels(formatted_labels)

        texts = [d.body for d in documents]
        titles = [d.title for d in documents]
        years: list[str] = [str(d.timestamp) for d in documents]

        # If documents are passed then those are embedded using the selected emb_model (else training emb are used)
        emb = self.__train_embeddings
        if not emb_train:
            emb = self._embedding_model.encode(texts, show_progress_bar=False)

        # Reduce dimensions for document visualization
        reduced_embeddings = self._reduction_model.transform(emb)

        fig_topics = self._topic_model.visualize_topics()
        fig_topics.write_html(path / "topic_space.html")
        fig_doc_topics = self._topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings, hide_annotations=True, custom_labels=True)
        fig_doc_topics.write_html(path / "document_clusters.html")

        topics_over_time = self._topic_model.topics_over_time(texts, years, nr_bins=20, datetime_format="%Y")
        fig_time = self._topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=None, custom_labels=True)
        fig_time.write_html(path / "topic_evolution.html")

        fig_hier = self._topic_model.visualize_hierarchy(top_n_topics=None, custom_labels=True)
        fig_hier.write_html(path / "topic_hierarchy.html")
