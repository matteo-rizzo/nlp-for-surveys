from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sized, Any

import numpy as np
import tqdm
import yaml
from gensim.models import KeyedVectors
from nltk import word_tokenize
from sklearn.cluster import MiniBatchKMeans
from spacy.lang.en import STOP_WORDS

from topic_extraction.BaseTopicExtractor import TopicExtractor
from topic_extraction.TextRank import TopicRank


def dump_yaml(path: str | Path, data: Any) -> None:
    """
    Load YAML as python dict

    :param path: path to YAML file
    :param data: data to dump

    :return: dictionary containing data
    """
    with open(path, encoding="UTF-8", mode="w") as f:
        yaml.dump(data, f, yaml.SafeDumper)


def preprocess(doc: str, max_words: int = 500) -> list[str]:
    rem_url = re.sub(r'http\S+', '', doc)
    rem_strange = re.sub(r'<[()]>@\\/', '', rem_url)
    rem_num = re.sub('[0-9]+', '', rem_strange)
    rem_spaces = re.sub('\s+', ' ', rem_num).strip()
    # tokenizer = RegexpTokenizer(r'\w+')
    tokens: list[str] = word_tokenize(rem_spaces)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in STOP_WORDS][:max_words]
    return filtered_words


def batch_list(iterable: Sized, batch_size: int = 10) -> Iterable:
    """
    Yields batches from an iterable container

    :param iterable: elements to be batched
    :param batch_size: (max) number of elements in a single batch
    :return: generator of batches
    """
    data_len = len(iterable)
    for ndx in range(0, data_len, batch_size):
        yield iterable[ndx:min(ndx + batch_size, data_len)]


class ClusteringMethod(TopicExtractor):
    def __init__(self):
        self._model = None
        self._embedding_vectors = None

    def __extract_features(self, documents):
        print("*** Start feature extraction ***")
        algo = TopicRank()
        algo.prepare()
        topics_rank: list[list[tuple[str, float]]] = algo.batch_extract(documents, k=5)
        important_terms: list[list[str]] = [[a[0] for a in l] for l in topics_rank]
        return important_terms

    def __compute_term_embedding(self, sentence: str) -> np.ndarray:
        tokens: list[str] = word_tokenize(sentence)
        return self._embedding_vectors.get_mean_vector(tokens, ignore_missing=True)

    def prepare(self, *args, **kwargs):
        # Load embeddings
        embs = kwargs.get("pre_trained_path", None)
        if embs:
            self._embedding_vectors = KeyedVectors.load_word2vec_format(embs)
        else:
            raise ValueError("Not implemented for noe")
        # Set up clustering
        self._model: MiniBatchKMeans = MiniBatchKMeans(init='k-means++', n_clusters=15)
        print("*** Loaded embeddings ***")

    def train(self, documents: list[str], *args, **kwargs):
        batch_size: int = kwargs.get("batch_size", 256)

        important_terms = self.__extract_features(documents)

        # Compute embedding features
        features_doc_batch: list[list[str]]
        for features_doc_batch in tqdm.tqdm(batch_list(important_terms, batch_size)):
            # tokenized_docs: list[list[str]] = [preprocess(doc) for doc in docs]
            # tokenized_docs: list[str] = [f.lower() for features_doc in features_doc_batch for f in features_doc if f.lower() in self._embedding_vectors.index_to_key]

            flattened_terms: list[str] = list()
            for features_doc in features_doc_batch:
                for f in features_doc:
                    if f in self._embedding_vectors.index_to_key:
                        flattened_terms.append(f)
                    elif f.lower() in self._embedding_vectors.index_to_key:
                        flattened_terms.append(f.lower())
            # flattened_terms: list[str] = [f.lower() for features_doc in features_doc_batch for f in features_doc]
            feature_embeddings: np.ndarray = np.stack([self.__compute_term_embedding(t) for t in flattened_terms])
            # embeddings: np.ndarray = self._embedding_vectors[tokenized_docs]
            # Set up clustering
            self._model.partial_fit(feature_embeddings)

        print("*** Finished clustering training ***")

    def extract(self, document: str, k: int, *args, **kwargs) -> list:
        pass

    def batch_extract(self, documents: list[str], k: int, *args, **kwargs) -> list[list]:
        batch_size: int = kwargs.get("batch_size", 16)

        important_terms = self.__extract_features(documents)
        flattened_terms: list[str] = [f.lower() for features_doc in important_terms for f in features_doc]
        feature_embeddings: np.ndarray = np.stack([self.__compute_term_embedding(t) for t in flattened_terms])

        # tokenized_docs: list[list[str]] = [preprocess(doc) for doc in documents]
        # tokenized_docs: list[str] = [b.lower() for a in tokenized_docs for b in a if b.lower() in self._embedding_vectors.index_to_key]
        # embeddings: np.ndarray = self._embedding_vectors[tokenized_docs]
        cluster_indices = self._model.predict(feature_embeddings)

        words_by_cluster = {}
        for i, idx in enumerate(cluster_indices.tolist()):
            ci_w = words_by_cluster.setdefault(idx, set())
            ci_w.add(flattened_terms[i])

        words_by_cluster = {k: list(v) for k, v in words_by_cluster.items()}
        dump_yaml("clusters.yml", words_by_cluster)
