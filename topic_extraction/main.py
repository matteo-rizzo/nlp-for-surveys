from pathlib import Path

import pandas as pd
import torch.cuda

from topic_extraction.classes.BERTopicExtractor import BERTopicExtractor
from topic_extraction.extraction import document_extraction
from topic_extraction.utils import dump_yaml, save_csv_results

pd.set_option("display.max_columns", None)

docs = document_extraction()

# Pass 1

# seed_topic_list = [
#     ["circular economy", "sustainability", "sustainable business", "sustainable development", "recycling", "waste"],
#     ["digitalization", "digital business", "digital economy", "digital innovation", "business transformation"]
# ]

seed_topic_list = [
    ["green", "sustainability", "environment", "transition", "transform", "business"],
    ["digitalization", "digital", "transition", "transform", "business"],
]

# --------------------- PASS 1
pl_path1 = Path("plots") / "themes"
ex1 = BERTopicExtractor(plot_path=pl_path1)
ex1.prepare(config_file="topic_extraction/config/bertopic1.yml", seed_topic_list=seed_topic_list)
ex1.train(docs)

# ex._topic_model.merge_topics([d.body for d in docs], topics_to_merge=[1, 2])
l1_topics, probs, l1_words_topics = ex1.batch_extract(docs, -1, use_training_embeddings=True)
torch.cuda.empty_cache()
# del ex1

# Plot/save results
# ex1.plot_wonders(docs)

l1_words = {k: [w for w, _ in ws] for k, ws in l1_words_topics.items()}
dump_yaml(l1_words, pl_path1 / "word_list.yml")

# --------------------- END PASS 1


# --------------------- PASS 2
# Determine field of application

pl_path2 = Path("plots") / "fields"
ex2 = BERTopicExtractor(plot_path=pl_path2)
ex2.prepare(config_file="topic_extraction/config/bertopic2.yml")
ex2.train(docs, normalize=True, embeddings=ex1._train_embeddings)
l2_topics, probs, l2_words_topics = ex2.batch_extract(docs, -1, use_training_embeddings=True)
# ex2.plot_wonders(docs, add_doc_classes=l1_topics)

l2_words = {k: [w for w, _ in ws] for k, ws in l2_words_topics.items()}
dump_yaml(l2_words, pl_path2 / "word_list.yml")

# --------------------- END PASS 2

save_csv_results(docs, themes=l1_topics, theme_keywords=l1_words, subjects=l2_topics, subj_keywords=l2_words, path=pl_path1.parent / "results")

# ex.see_topic_evolution(docs, bins_n=3)
