from __future__ import annotations

import copy
import datetime
import logging
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
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

from topic_extraction.visualization.plotly_graph import plot_network
from topic_extraction.visualization.utils import visualize_topic_space_data


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
        self._plot_path: Path = Path("plots")
        self._instantiation_kwargs = None

    @staticmethod
    def tl_factory(tl_args: dict) -> BERTopic:
        return BERTopic(**tl_args)

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
            tl_args[
                "representation_model"] = self._representation_model  # Step 6 - (Optional) Fine-tune topic representations

        self._instantiation_kwargs = {
            **tl_args,
            **model_config["bertopic"]
        }
        self._topic_model = BERTopicExtractor.tl_factory(self._instantiation_kwargs)

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

        g: nx.DiGraph = nx.DiGraph()
        k = 2

        # timestamps: list[int] = [d.timestamp for d in documents]
        timestamps: list[int] = list(range(len(documents)))

        # Create bins of timestamps
        # bins = np.linspace(min(timestamps), max(timestamps), bins_n + 1)

        # Try to select bin edges based on frequency
        n_samples: int = len(timestamps) // bins_n
        bins = np.sort(timestamps).astype(float)[list(range(0, len(timestamps), n_samples))]

        # Assign documents to bins
        doc_bins = np.digitize(timestamps, bins)

        # Extract text for each bin
        doc_by_bin: dict[int, list[str]] = {}
        for doc, idx in zip(documents, doc_bins.tolist()):
            docs = doc_by_bin.setdefault(idx, list())
            docs.append(doc.body)

        # Merge bins with less than n_samples
        merge_bins = [idx for idx, docs in doc_by_bin.items() if len(docs) < n_samples]
        for bin_ in merge_bins:
            if bin_ + 1 in doc_by_bin:
                doc_by_bin[bin_ + 1].extend(doc_by_bin[bin_])
            else:
                doc_by_bin[bin_ - 1].extend(doc_by_bin[bin_])
            del doc_by_bin[bin_]

        umap_final: UMAP = UMAP(n_neighbors=2, n_components=2, init="random", metric="hellinger", random_state=1561)
        umap_final.fitted = False

        embedding_model = SentenceTransformer(self._config["model"]["sentence_transformer"])
        umap_model = UMAP(**self._config["model"]["dimensionality_reduction"]["params"]["umap"])
        ctfidf_model = ClassTfidfTransformer(**self._config["model"]["weighting"]["params"])

        conf = copy.deepcopy(self._instantiation_kwargs)
        conf["umap_model"] = umap_model
        conf["embedding_model"] = embedding_model
        conf["ctfidf_model"] = ctfidf_model

        model_prev = None
        g_prev_topics_ids: list[str] = list()
        # Clustering
        for idx, docs in doc_by_bin.items():
            # New BERTopic instance
            t_model = BERTopicExtractor.tl_factory(conf)
            # embeddings = self._embedding_model.encode(docs)

            # t_model.fit(docs, embeddings=embeddings)
            t_model.fit(docs)

            print("**** NEW BIN ****")
            new_nodes = list()

            # names: list[str] = t_model.get_topic_info()["Name"].tolist()[1:]
            t_names: list[str] = \
                t_model.generate_topic_labels(nr_words=4, topic_prefix=True, word_length=None, separator=" - ")[1:]

            n_topics = len(t_names)
            if n_topics < 3:
                logging.warning(f"Found {n_topics}. Skipping this bin.")
                continue
            if n_topics < k:
                logging.warning(f"Insufficient number of topics, reducing k to {n_topics}")
                k = n_topics
            df = visualize_topic_space_data(t_model, umap_final)
            assert df["x"].size == df["y"].size == len(t_names), f"Wrong {df['x'].size} {df['y'].size} {len(t_names)}"

            # t_names = df["Words"].tolist()
            sizes = df["Size"].tolist()

            if model_prev is not None:
                # print(t_model.get_topic_info()["Name"].tolist())
                # print(len(model_prev.topic_embeddings_))
                # print(names)  # (n_new_topics, )
                sim_matrix = cosine_similarity(model_prev.topic_embeddings_, t_model.topic_embeddings_)
                # print(sim_matrix.shape)  # (n_old_topics, n_new_topics)

                for prev_t in g_prev_topics_ids:
                    # Get topic model ID for each previous topic
                    t_id: int = g.nodes[prev_t]["ID"]  # topic model ID

                    # Compute similarity with new topics (get top-k most similar topics)
                    most_similar_topics: list[int] = (np.argpartition(sim_matrix[t_id + 1], -k)[-k:] - 1).tolist()
                    # If -1 (outliers) is found to be similar, remove it
                    most_similar_topics = [s for s in most_similar_topics if s >= 0]

                    assert len(t_names) > max(
                        most_similar_topics), f"Topics: {most_similar_topics}, names: {len(t_names)}"
                    # assert min(most_similar_topics) >= 0, f"-1 is a similar topic"

                    # Add new nodes with new topic information
                    for new_t in most_similar_topics:
                        g_id = f"{idx}_{new_t}"
                        if g_id not in g:
                            g.add_node(g_id, ID=new_t, name=t_names[new_t],
                                       pos=(df["x"][new_t], df["y"][new_t]), size=sizes[new_t])  # TODO: add topic info
                            new_nodes.append(g_id)
                        # Add weighted edges to represent similarity
                        g.add_edge(prev_t, g_id, w=float(sim_matrix[t_id + 1, new_t]))
            else:
                for i, n in enumerate(t_names):
                    g_id = f"{idx}_{i}"
                    cluster_coords = (df["x"][i], df["y"][i])
                    g.add_node(g_id, ID=i, name=n, pos=cluster_coords, size=sizes[i])  # TODO: add topic info
                    new_nodes.append(g_id)

            model_prev = t_model
            g_prev_topics_ids = new_nodes

        # Plot network
        nx.draw(g)
        plt.show()
        fig = plot_network(g)
        self._plot_path.mkdir(parents=True, exist_ok=True)
        fig.write_html(self._plot_path / "topic_time_network.html")

        # if model_prev is None:
        #     # initialize topics
        #     names = t_model.get_topic_info()["Name"].tolist()[1:]
        #     print(names)
        #     next_topic = [
        #         [(int(k), names[k])] for k in t_model.get_topics().keys()
        #     ]
        # else:
        #     sim_matrix = cosine_similarity(model_prev.topic_embeddings_, t_model.topic_embeddings_)
        #     for i in range(len(next_topic)):
        #         last_topic_entry: int = next_topic[i][-1][0]
        #         most_similar_topic: int = np.argmax(sim_matrix[last_topic_entry + 1]) - 1
        #         names = t_model.get_topic_info()["Name"].tolist()[1:]
        #         print(names)
        #         print(most_similar_topic)
        #         next_topic[i].append((most_similar_topic, names[most_similar_topic]))
        #
        # model_prev = t_model

        # Print topics to file
        # pprint(next_topic)

    def plot_wonders(self, documents: list[Document], **kwargs) -> None:

        print("*** Plotting results ***")

        emb_train: bool = kwargs.get("use_training_embeddings", False)

        self._plot_path.mkdir(parents=True, exist_ok=True)

        formatted_labels = self._topic_model.generate_topic_labels(nr_words=4, topic_prefix=False, word_length=None,
                                                                   separator=" - ")
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
        fig_topics.write_html(self._plot_path / "topic_space.html")
        fig_doc_topics = self._topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings,
                                                               hide_annotations=True, custom_labels=True)
        fig_doc_topics.write_html(self._plot_path / "document_clusters.html")

        topics_over_time = self._topic_model.topics_over_time(texts, years, nr_bins=20, datetime_format="%Y")
        fig_time = self._topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=None, custom_labels=True,
                                                                normalize_frequency=True)
        fig_time.write_html(self._plot_path / "topic_evolution.html")

        fig_hier = self._topic_model.visualize_hierarchy(top_n_topics=None, custom_labels=True)
        fig_hier.write_html(self._plot_path / "topic_hierarchy.html")
