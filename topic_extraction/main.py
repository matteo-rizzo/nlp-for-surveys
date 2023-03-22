from pathlib import Path

import pandas as pd

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
ex.plot_wonders(docs)
dump_yaml(words_topics, Path("plots") / "word_list.yml")

ex.see_topic_evolution(docs, bins_n=3)
