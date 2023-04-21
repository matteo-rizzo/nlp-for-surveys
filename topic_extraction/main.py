from pathlib import Path

import pandas as pd

from topic_extraction.classes.BERTopicExtractor import BERTopicExtractor

from topic_extraction.extraction import document_extraction
from topic_extraction.utils import dump_yaml

pd.set_option('display.max_columns', None)

docs = document_extraction()

# Pass 1

seed_topic_list = [
    ["circular economy", "sustainability", "sustainable business", "sustainable development", "recycling", "waste"],
    ["digitalization", "digital business", "digital economy", "digital innovation", "business transformation"]
]

ex = BERTopicExtractor()
ex.prepare(config_file="topic_extraction/config/bertopic1.yml", seed_topic_list=seed_topic_list)

# ex.tuning(docs)

ex.train(docs)
topics, probs, words_topics = ex.batch_extract(docs, -1, use_training_embeddings=True)

# Plot/save results
words_topics = {k: [w for w, _ in ws] for k, ws in words_topics.items()}
ex.plot_wonders(docs)
dump_yaml(words_topics, Path("plots") / "word_list.yml")

# Pass 2
# Determine field of application
# TODO


# ex.see_topic_evolution(docs, bins_n=3)
