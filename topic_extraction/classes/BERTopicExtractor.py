from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path
from typing import Callable, Generic, TypeVar, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from topictuner import TopicModelTuner as TMT
from umap import UMAP

from topic_extraction.classes import Document
from topic_extraction.classes.BERTopicExtended import BERTopicExtended
from topic_extraction.classes.BaseTopicExtractor import BaseTopicExtractor
from topic_extraction.utils import load_yaml
from topic_extraction.visualization.visualize_stacked_topics import visualize_stacked_topics

T = TypeVar("T")

logger = logging.getLogger(__name__)


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

        self._embedding_save_path = f'dumps/embeddings/{model_config["sentence_transformer"]}.npy'
        Path(self._embedding_save_path).parent.mkdir(exist_ok=True, parents=True)

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
        fit_reduction = kwargs.get("fit_reduction", True)

        if embeddings is None:
            # Precompute embeddings
            embeddings = self._embedding_model.encode(texts, show_progress_bar=False)
            if kwargs.get("normalize", False):
                # NOTE: only works when batch_extract use the training embeddings
                embeddings /= np.linalg.norm(embeddings, axis=1).reshape(-1, 1)

            np.save(self._embedding_save_path, embeddings)

        self._train_embeddings = embeddings

        print("*** Fitting the model ***")

        # Topic modelling
        # topics, probs = \
        self._topic_model.fit(texts, embeddings=embeddings, fit_reduction=fit_reduction)
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

        # Use pre-trained /reduced embeddings.
        emb = self._train_embeddings if emb_train else None

        topics, probs = self._topic_model.transform(texts, embeddings=emb)

        print(f"Outliers: {len([t for t in topics if t < 0])}")

        if self._reduce_outliers:
            print("*** Reducing outliers ***")
            topics = self._topic_model.reduce_outliers(texts, topics)
            print(f"Outliers post-reduction: {len([t for t in topics if t < 0])}")

        return topics, probs, self._topic_model.get_topics()

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

        # topics_over_time = self._topic_model.topics_over_time(texts, years, nr_bins=20, datetime_format="%Y")
        # fig_time = self._topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=None, custom_labels=True,
        #                                                         normalize_frequency=False, relative_frequency=True, width=1600, height=800)
        # fig_time.write_html(self._plot_path / "topic_evolution.html")

        # fig_hier = self._topic_model.visualize_hierarchy(top_n_topics=None, custom_labels=True)
        # fig_hier.write_html(self._plot_path / "topic_hierarchy.html")

        # topics_per_class = self._topic_model.topics_per_class(texts, classes=GROUND_TRUTH)
        # fig_class = self._topic_model.visualize_topics_per_class(custom_labels=True)
        # fig_class.write_html(self._plot_path / "topic_hierarchy.html")

        if kwargs.get("add_doc_classes", None):
            l2_topics = kwargs["add_doc_classes"]
            fig = visualize_stacked_topics(self._topic_model, titles, reduced_embeddings=reduced_embeddings, hide_annotations=True, custom_labels=True, width=1800, height=1200,
                                           stacked_topics=l2_topics, stacked_symbols=[(0, "circle"), (1, "x")])
            fig.write_html(self._plot_path / "topic_stacked.html")
