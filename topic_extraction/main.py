from pathlib import Path

import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer

from topic_extraction.classes.BERTopicExtractor import BERTopicExtractor

from topic_extraction.extraction import document_extraction
from topic_extraction.utils import dump_yaml

pd.set_option('display.max_columns', None)

ex = BERTopicExtractor()
ex.prepare(config_file="topic_extraction/config/bertopic.yml")
docs = document_extraction()
ex.train(docs)
topics, probs, words_topics = ex.batch_extract(docs, -1, use_training_embeddings=True)

# Plot/save results
words_topics = {k: [w for w, _ in ws] for k, ws in words_topics.items()}
dump_yaml(words_topics, Path("plots") / "word_list.yml")
ex.plot_wonders(docs)

# Step 1 - Extract embeddings
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#
# # Step 2 - Reduce dimensionality
# umap_model = UMAP(n_neighbors=20, n_components=5, min_dist=0.0, metric='cosine', low_memory=False)
#
# # Step 3 - Cluster reduced embeddings
# hdbscan_model = HDBSCAN(min_cluster_size=20, min_samples=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
#
# # is UMAP.n_components is increased may want to change metric in HDBSCAN
#
# # Step 4 - Tokenize topics
# vectorizer_model = CountVectorizer(stop_words=["english", "french"])
#
# # Step 5 - Create topic representation
# ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
#
# # Step 6 - (Optional) Fine-tune topic representations with
# # a `bertopic.representation` model
# # representation_model = KeyBERTInspired()
# representation_model = MaximalMarginalRelevance(diversity=0.2)
#
# texts, titles, years, keywords = text_extraction()
# # keyword_set = {k for doc_keywords in keywords for k in doc_keywords}
#
# # All steps together
# topic_model = BERTopic(
#     embedding_model=embedding_model,  # Step 1 - Extract embeddings
#     umap_model=umap_model,  # Step 2 - Reduce dimensionality
#     hdbscan_model=hdbscan_model,  # Step 3 - Cluster reduced embeddings
#     vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
#     ctfidf_model=ctfidf_model,  # Step 5 - Extract topic words
#     representation_model=representation_model,  # Step 6 - (Optional) Fine-tune topic representations
#     # seed_topic_list= keyword_set
#     min_topic_size=15,
#     language="multilingual"
# )
#
# # Precompute embeddings
# embeddings = embedding_model.encode(texts, show_progress_bar=False)
#
# # Topic modelling
# topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)
# formatted_labels = topic_model.generate_topic_labels(nr_words=4, topic_prefix=False, word_length=16, separator=" - ")
# topic_model.set_topic_labels(formatted_labels)
# print(topic_model.get_topic_info())
#
# # Get document embeddings
# # Reduce dimensions for document visualization
# reduced_embeddings = umap_model.transform(embeddings)
#
# fig_topics = topic_model.visualize_topics()
# fig_topics.write_html("plots/topic.html")
# fig_doc_topics = topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings, hide_annotations=True, custom_labels=True)
# fig_doc_topics.write_html("plots/doc_topic.html")
#
# topics_over_time = topic_model.topics_over_time(texts, years, nr_bins=20, datetime_format="%Y")
# fig_time = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=None, custom_labels=True)
# fig_time.write_html("plots/time_topic.html")
