from __future__ import annotations

import copy
import itertools
import logging
from pathlib import Path
from typing import Callable, Generic, TypeVar, Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from topictuner import TopicModelTuner as TMT
from umap import UMAP

from topic_extraction.classes import Document
from topic_extraction.classes.BERTopicExtended import BERTopicExtended
from topic_extraction.classes.BaseTopicExtractor import BaseTopicExtractor
from topic_extraction.utils import load_yaml
from topic_extraction.visualization.plotly_graph import plot_network
from topic_extraction.visualization.utils import visualize_topic_space_data
from topic_extraction.visualization.visualize_stacked_topics import visualize_stacked_topics

T = TypeVar("T")


def grid_search(estimator_class: T, grid_params: dict, metric_fun: Callable[[Generic[T]], float], large_is_better: bool,
                estimator_fit_args: Iterable = None, estimator_fit_kwargs: dict = None, estimator_kwargs: dict = None) -> list[dict]:
    print("Started grid search...")

    estimator_fit_args = estimator_fit_args if estimator_fit_args is not None else list()
    estimator_fit_kwargs = estimator_fit_kwargs if estimator_fit_kwargs is not None else dict()
    estimator_kwargs = estimator_kwargs if estimator_kwargs is not None else dict()

    best_score = -1.0
    best_parameters = None
    all_results = list()

    compare_fn = float.__gt__ if large_is_better else float.__lt__

    k_args, args_list = zip(*grid_params.items())
    args_len = [list(range(len(a))) for a in args_list]
    args_index_combinations: list[tuple] = itertools.product(*args_len)

    for comb_idx in args_index_combinations:
        # Prepare argument combination
        vals = [arg[i] for arg, i in zip(args_list, comb_idx)]
        kv_args = dict(zip(k_args, vals))

        # Clustering
        est = estimator_class(**kv_args, **estimator_kwargs).fit(*estimator_fit_args, **estimator_fit_kwargs)
        # DBCV score
        score = metric_fun(est)
        n_clusters = int(est.labels_.max() + 1)
        ext_args = {**kv_args, "score": score, "n_clusters": n_clusters}
        # if we got a better score, store it and the parameters
        if compare_fn(score, best_score):
            best_score = score
            best_parameters = ext_args
        all_results.append(ext_args)

    print("Best score: {:.4f}".format(best_score))
    print("Best parameters: {}".format(best_parameters))
    return all_results


class BERTopicExtractor(BaseTopicExtractor):
    def save(self, path: str | Path, *args, **kwargs):
        self._topic_model.save(path, *args, **kwargs)

    def load(self, path: str | Path, *args, **kwargs):
        self._topic_model = BERTopic.load(path, *args, **kwargs)
        # TODO: assign components

    def __init__(self, plot_path: Path | str = Path("plots")):
        self._train_embeddings = None
        self._topic_model: BERTopicExtended = None
        self._reduction_model = None
        self._config = None
        self._embedding_model = None
        self._clustering_model = None
        self._vectorizer_model = None
        self._weighting_model = None
        self._representation_model = None
        self._reduce_outliers: bool = False
        self._plot_path: Path = plot_path
        self._instantiation_kwargs = None

    @staticmethod
    def tl_factory(tl_args: dict) -> BERTopic:
        return BERTopicExtended(**tl_args)

    def find_optimal_n_clusters(self, documents, conf_search: str | Path, **kwargs) -> None:

        conf_search = load_yaml(conf_search)["clustering"]

        documents = [d.body for d in documents]

        embeddings = self._embedding_model.encode(documents, show_progress_bar=False)

        if kwargs.get("normalize", False):
            embeddings /= np.linalg.norm(embeddings, axis=1).reshape(-1, 1)

        result_path = "hdbscan_grid_results"
        if kwargs.get("result_path", False):
            result_path = kwargs["result_path"]

        umap_embeddings = self._topic_model._reduce_dimensionality(embeddings)

        # 2. Select best hyperparameters

        if isinstance(self._clustering_model, KMeans):
            c = conf_search["kmeans"]
            distortions, silhouette_scores = [], []
            k_range = range(c["k_start"], c["k_end"])
            for k in k_range:
                clustering = KMeans(n_clusters=k, n_init="auto", random_state=0)
                clustering.fit(umap_embeddings)
                distortions.append(clustering.inertia_)  # lower the better
                silhouette_scores.append(silhouette_score(umap_embeddings, clustering.labels_))  # higher the better (ideal > .5)

            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(k_range, distortions, "bx-")
            ax[0].set_xlabel("Number of Clusters")
            ax[0].set_ylabel("Distortion (the lower the better)")
            ax[0].set_title("Elbow Method")
            ax[0].grid(True)

            ax[1].plot(k_range, silhouette_scores, "bx-")
            ax[1].set_xlabel("Number of Clusters")
            ax[1].set_ylabel("Silhouette Score (the higher the better)")
            ax[1].set_title("Silhouette Method")
            ax[1].grid(True)

            plt.show()

            optimal_n_clusters = np.argmax(silhouette_scores) + c["k_start"]
        else:
            # logging.captureWarnings(True)
            # parameters and distributions to sample from
            c = conf_search["hdbscan"]

            # Define the score function
            def fun_dbcv(est: HDBSCAN) -> float:
                return float(est.relative_validity_)

            results = grid_search(HDBSCAN, grid_params=c, metric_fun=fun_dbcv, estimator_fit_args=(umap_embeddings,), large_is_better=True,
                                  estimator_kwargs=dict(gen_min_span_tree=True))

            dfr = pd.DataFrame.from_records(results).sort_values(by="score", ascending=False)

            self._plot_path.mkdir(exist_ok=True, parents=True)
            dfr.to_csv(self._plot_path / f"{result_path}.csv")

            optimal_n_clusters = dfr["n_clusters"][0]

        print(f"The optimal number of clusters is {optimal_n_clusters}")

    def prepare(self, *args, **kwargs):
        config_path: str | Path = kwargs.pop("config_file")
        self._config = load_yaml(config_path)

        run_config = self._config["run"]
        model_config = self._config["model"]

        print("*** Preparing everything ***")

        # Step 1 - Extract embeddings
        self._embedding_model = SentenceTransformer(model_config["sentence_transformer"])

        # Step 2 - Reduce dimensionality
        model_rd = kwargs.pop("dimensionality_reduction", None)
        if not model_rd:
            conf = model_config["dimensionality_reduction"]
            if conf["choice"] == "umap":
                model_rd = UMAP(**conf["params"][conf["choice"]])
            elif conf["choice"] == "pca":
                model_rd = PCA(**conf["params"][conf["choice"]])
        self._reduction_model = model_rd

        # Step 3 - Cluster reduced embeddings
        model_cl = kwargs.pop("clustering", None)
        if not model_cl:
            conf = model_config["clustering"]
            model_cl = None
            if conf["choice"] == "hdbscan":
                model_cl = HDBSCAN(**conf["params"][conf["choice"]])
            elif conf["choice"] == "kmeans":
                model_cl = KMeans(**conf["params"][conf["choice"]])
        self._clustering_model = model_cl
        # if UMAP.n_components is increased may want to change metric in HDBSCAN

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
            **model_config["bertopic"],
            **kwargs
        }
        self._topic_model = BERTopicExtractor.tl_factory(self._instantiation_kwargs)

        self._reduce_outliers = run_config["reduce_outliers"]

    def tuning(self, documents: list[Document]):
        texts = [d.body for d in documents]

        print("*** Tuning BERTopic ***")

        tmt = TMT(verbose=0)

        tmt.createEmbeddings(texts)  # Run embedding model
        tmt.reduce()  # Run UMAP
        # lastRunResultsDF = tmt.randomSearch([*range(10, 200)], [.1, .25, .5, .75, 1], iters=50)

        lastRunResultsDF = tmt.gridSearch([*range(35, 41)])  # [x / 100 for x in range(10, 101, 10)]
        summaryDF = tmt.summarizeResults(lastRunResultsDF).sort_values(by=["number_uncategorized"])
        # tmt.visualizeSearch(lastRunResultsDF).show()
        tmt.visualizeSearch(lastRunResultsDF).show()
        summaryDF.to_csv("tuning3.csv")
        print(lastRunResultsDF)

    def train(self, documents: list[Document], *args, **kwargs) -> None:
        texts = [d.body for d in documents]

        print("*** Generating embeddings ***")

        embeddings = kwargs.get("embeddings", None)

        if embeddings is None:
            # Precompute embeddings
            embeddings = self._embedding_model.encode(texts, show_progress_bar=False)
            if kwargs.get("normalize", False):
                # NOTE: only works when batch_extract use the training embeddings
                embeddings /= np.linalg.norm(embeddings, axis=1).reshape(-1, 1)
        self._train_embeddings = embeddings

        print("*** Fitting the model ***")

        # Topic modelling
        # topics, probs = \
        self._topic_model.fit(texts, embeddings=embeddings)
        # Further reduce topics
        # self._topic_model.reduce_topics(texts, nr_topics=3)

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

        emb = self._train_embeddings if emb_train else None

        topics, probs = self._topic_model.transform(texts, embeddings=emb)

        print(f"Outliers: {len([t for t in topics if t < 0])}")

        if self._reduce_outliers:
            print("*** Reducing outliers ***")
            topics = self._topic_model.reduce_outliers(texts, topics)
            print(f"Outliers post-reduction: {len([t for t in topics if t < 0])}")

        return topics, probs, self._topic_model.get_topics()

    def see_topic_evolution(self, documents: list[Document], bins_n: int, k: int = 1, min_samples: int = 80):

        g: nx.DiGraph = nx.DiGraph()
        k_ = k

        timestamps: list[int] = [d.timestamp for d in documents]
        # timestamps: list[int] = list(range(len(documents)))

        # Create bins of timestamps
        # bins_time = np.linspace(min(timestamps), max(timestamps), bins_n + 1)

        # Try to select bin edges based on frequency
        # n_samples: int = len(timestamps) // bins_n
        # bins_time = np.sort(timestamps).astype(float)[list(range(0, len(timestamps), n_samples))]
        # Find ~equal-sized bins
        o, bins_time = pd.qcut(timestamps, bins_n, retbins=True)

        # Assign documents to bins
        bins_time_ = [bins_time[0] - 0.1, *bins_time[1:-1], bins_time[-1] + 0.1]
        doc_bins = np.digitize(timestamps, bins_time_, right=True)

        # Extract text for each bin
        doc_by_bin: dict[int, list[str]] = dict()
        for doc, idx in zip(documents, doc_bins.tolist()):
            docs = doc_by_bin.setdefault(idx, list())
            docs.append(doc.body)

        # Merge bins with less than n_samples
        merge_bins = [idx for idx, docs in doc_by_bin.items() if len(docs) < min_samples]
        for bin_ in merge_bins:
            if bin_ + 1 in doc_by_bin:
                doc_by_bin[bin_ + 1].extend(doc_by_bin[bin_])
            else:
                doc_by_bin[bin_ - 1].extend(doc_by_bin[bin_])
            del doc_by_bin[bin_]

        assert 0 not in doc_by_bin, "Bin 0 is present in doc_by_bin. This is unexpected :("

        bins = [int(min(timestamps))] + list(sorted([int(bins_time[a]) for a in doc_by_bin.keys()]))

        all_docs = [d.body for d in documents]

        vec_conf = dict()  # self._config["model"]["vectorizer"]["params"]
        vectorizer: CountVectorizer = CountVectorizer(**vec_conf).fit(all_docs)
        vocab = vectorizer.get_feature_names_out()
        vectorizer = CountVectorizer(**vec_conf, vocabulary=vocab)

        embedding_model = SentenceTransformer(self._config["model"]["sentence_transformer"])
        all_embeddings = embedding_model.encode(all_docs, show_progress_bar=False)

        umap_final: UMAP = UMAP(n_neighbors=2, n_components=2, init="random", metric="cosine", random_state=1561)
        umap_final.fit(all_embeddings)
        umap_final.fitted = True

        umap_model = UMAP(**self._config["model"]["dimensionality_reduction"]["params"]["umap"])

        conf = copy.deepcopy(self._instantiation_kwargs)
        conf["umap_model"] = umap_model
        # conf["embedding_model"] = embedding_model
        conf["vectorizer_model"] = vectorizer

        model_prev = None
        g_prev_topics_ids: list[str] = list()
        # Clustering
        for idx, docs in sorted(doc_by_bin.items()):
            # offset: int = 0 if model_prev is None else 1
            bin_name = f"( {bins[idx - 1]}-{bins[idx]} ]"
            # New BERTopic instance
            t_model = BERTopicExtractor.tl_factory(conf)
            embeddings = embedding_model.encode(docs, show_progress_bar=False)

            t_model = t_model.fit(docs, embeddings=embeddings)
            # t_model.fit(docs)

            print("**** NEW BIN ****")
            new_nodes = list()

            # names: list[str] = t_model.get_topic_info()["Name"].tolist()[1:]
            t_names: list[str] = \
                t_model.generate_topic_labels(nr_words=4, topic_prefix=True, word_length=None, separator=" - ")[1:]

            n_topics = len(t_names)
            if n_topics < 3:
                logging.warning(f"Found {n_topics}. Skipping this bin.")
                continue
            if n_topics < k_:
                logging.warning(f"Insufficient number of topics, reducing k to {n_topics}")
                k_ = n_topics
            df = visualize_topic_space_data(t_model, umap_final)
            assert df["x"].size == df["y"].size == len(t_names), f"Wrong {df['x'].size} {df['y'].size} {len(t_names)}"

            t_names = df["Words"].tolist()
            t_sizes = df["Size"].tolist()

            if model_prev is not None:
                sim_matrix = cosine_similarity(model_prev.topic_embeddings_, t_model.topic_embeddings_)[1:, 1:]
                # print(sim_matrix.shape)  # (n_old_topics, n_new_topics)

                for prev_t in g_prev_topics_ids:
                    # Get topic model ID for each previous topic
                    t_id: int = g.nodes[prev_t]["ID"]  # topic model ID

                    # Compute similarity with new topics (get top-k most similar topics)
                    most_similar_topics: list[int] = (np.argpartition(sim_matrix[t_id], -k_)[-k_:]).tolist()
                    # If -1 (outliers) is found to be similar, remove it
                    most_similar_topics = [s for s in most_similar_topics if s >= 0]
                    if not most_similar_topics:
                        continue

                    assert len(t_names) > max(
                        most_similar_topics), f"Topics: {most_similar_topics}, names: {len(t_names)}"
                    # assert min(most_similar_topics) >= 0, f"-1 is a similar topic"

                    # Add new nodes with new topic information
                    for new_t in most_similar_topics:
                        g_id = f"{idx}_{new_t}"
                        if g_id not in g:
                            g.add_node(g_id, ID=new_t, name=t_names[new_t],
                                       pos=(df["x"][new_t], df["y"][new_t]), size=t_sizes[new_t], bin=bin_name)
                            new_nodes.append(g_id)
                        # Add weighted edges to represent similarity
                        g.add_edge(prev_t, g_id, w=float(sim_matrix[t_id, new_t]))
            else:
                for i, n in enumerate(t_names):
                    g_id = f"{idx}_{i}"
                    cluster_coords = (df["x"][i], df["y"][i])
                    g.add_node(g_id, ID=i, name=n, pos=cluster_coords, size=t_sizes[i], bin=bin_name)
                    new_nodes.append(g_id)

            model_prev = t_model
            g_prev_topics_ids = new_nodes

        # Plot network
        # nx.draw_networkx(g)
        # plt.show()
        fig = plot_network(g, width=1200, height=1200)
        self._plot_path.mkdir(parents=True, exist_ok=True)
        fig.write_html(self._plot_path / "topic_time_network.html")

    def plot_wonders(self, documents: list[Document], **kwargs) -> pd.DataFrame:

        print("*** Plotting results ***")

        emb_train: bool = kwargs.get("use_training_embeddings", False)

        self._plot_path.mkdir(parents=True, exist_ok=True)

        formatted_labels = self._topic_model.generate_topic_labels(nr_words=6, topic_prefix=False, word_length=None,
                                                                   separator=" - ")
        self._topic_model.set_topic_labels(formatted_labels)

        texts = [d.body for d in documents]
        titles = [f"{d.id} - {d.title}" for d in documents]
        years: list[str] = [str(d.timestamp) for d in documents]

        # If documents are passed then those are embedded using the selected emb_model (else training emb are used)
        emb = self._train_embeddings
        if not emb_train:
            emb = self._embedding_model.encode(texts, show_progress_bar=False)

        # Reduce dimensions for document visualization
        reduced_embeddings = self._reduction_model.transform(emb)

        fig_topics = self._topic_model.visualize_topics(width=1200, height=1200)
        fig_topics.write_html(self._plot_path / "topic_space.html")
        fig_doc_topics = self._topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings,
                                                               hide_annotations=True, custom_labels=True, width=1800, height=1200)
        fig_doc_topics.write_html(self._plot_path / "document_clusters.html")

        topics_over_time = self._topic_model.topics_over_time(texts, years, nr_bins=20, datetime_format="%Y")
        fig_time = self._topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=None, custom_labels=True,
                                                                normalize_frequency=False, relative_frequency=True, width=1600, height=800)
        fig_time.write_html(self._plot_path / "topic_evolution.html")

        fig_hier = self._topic_model.visualize_hierarchy(top_n_topics=None, custom_labels=True)
        fig_hier.write_html(self._plot_path / "topic_hierarchy.html")

        if kwargs.get("add_doc_classes", None):
            l2_topics = kwargs["add_doc_classes"]
            fig = visualize_stacked_topics(self._topic_model, titles, reduced_embeddings=reduced_embeddings, hide_annotations=True, custom_labels=True, width=1800, height=1200,
                                           stacked_topics=l2_topics, stacked_symbols=[(0, "circle"), (1, "x")])
            fig.write_html(self._plot_path / "topic_stacked.html")
        return topics_over_time
