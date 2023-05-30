from pathlib import Path

import numpy as np
import pandas as pd
import torch.cuda

from topic_extraction.classes.BERTopicExtractor import BERTopicExtractor
from topic_extraction.extraction import document_extraction
from topic_extraction.utils import dump_yaml, save_csv_results

pd.set_option("display.max_columns", None)

docs = document_extraction()

# Pass 1


seed_topic_list = [
    ["green", "sustainability", "environment", "transition", "transform", "business"],
    ["digitalization", "digital", "transition", "transform", "business"],
]

# --------------------- PASS 1
pl_path1 = Path("plots") / "themes"
pl_path1.mkdir(exist_ok=True, parents=True)
ex1 = BERTopicExtractor(plot_path=pl_path1)
ex1.prepare(config_file="topic_extraction/config/bertopic1.yml", seed_topic_list=seed_topic_list)

embeddings = None
if Path(ex1._embedding_save_path).is_file():
    embeddings = np.load(ex1._embedding_save_path)

ex1.train(docs, embeddings=embeddings)

# ex._topic_model.merge_topics([d.body for d in docs], topics_to_merge=[1, 2])
l1_topics, probs, l1_words_topics = ex1.batch_extract(docs, -1, use_training_embeddings=True)
del ex1
torch.cuda.empty_cache()

# Plot/save results
# ex1.plot_wonders(docs)

l1_words = {k: [w for w, _ in ws] for k, ws in l1_words_topics.items()}
dump_yaml(l1_words, pl_path1 / "word_list.yml")

# --------------------- END PASS 1


# --------------------- PASS 2
# Determine field of application

pl_path2 = Path("plots") / "fields"
pl_path2.mkdir(exist_ok=True, parents=True)
ex2 = BERTopicExtractor(plot_path=pl_path2)
ex2.prepare(config_file="topic_extraction/config/bertopic2.yml")
ex2.train(docs, embeddings=embeddings)
l2_topics, probs, l2_words_topics = ex2.batch_extract(docs, -1, use_training_embeddings=True, reduce_outliers=True, threshold=.5)
topic_over_time = ex2.plot_wonders(docs, add_doc_classes=l1_topics, use_training_embeddings=True)
l2_topics_all = ex2._topic_model.reduce_outliers([d.body for d in docs], l2_topics, probabilities=probs, strategy="probabilities", threshold=.3)

l2_words = {k: [w for w, _ in ws] for k, ws in l2_words_topics.items()}
dump_yaml(l2_words, pl_path2 / "word_list.yml")

# --------------------- END PASS 2


# --------------------- PASS 3

seed_topic_list2 = [
    ['fish', 'harvest', 'agro-food', 'agri-food', 'agrotourism', 'agro-chemical', 'horticulture', 'agriculture', 'agroecology', 'husbandry', 'agrifood', 'agribusiness',
     'agrochemical', 'farmer', 'bier', 'agro-industry', 'agroforestry', 'farm', 'farmland', 'aquaculture', 'crop growing', 'farmwork', 'agri-business', 'agroindustry',
     'sharecropping', 'agricultural', 'wine', 'cultivation', 'viticulture', 'beer', 'hydroponics', 'agrofood', 'food', 'farming', 'agronomy', 'livestock', 'agritourism',
     'agrifood-tech',
     "agri-food system", "agri-food ecosystem", "agri-food firm", "food system", "bio-district", "digital transformation in agriculture", "food value chain",
     "sustainable agriculture", "forest"]
]

pl_path3 = Path("plots") / "agrifood"
pl_path3.mkdir(exist_ok=True, parents=True)
ex3 = BERTopicExtractor(plot_path=pl_path3)
ex3.prepare(config_file="topic_extraction/config/bertopic3.yml", seed_topic_list=seed_topic_list2)  # dimensionality_reduction=ex2._reduction_model
del ex2._embedding_model
torch.cuda.empty_cache()

embeddings = None
if Path(ex3._embedding_save_path).is_file():
    embeddings = np.load(ex3._embedding_save_path)

ex3.train(docs, normalize=True, embeddings=embeddings, fit_reduction=True)
print(f"DBCV: {ex3._topic_model.hdbscan_model.relative_validity_}")
l3_topics, probs, l3_words_topics = ex3.batch_extract(docs, -1, use_training_embeddings=True)
l3_words = {k: [w for w, _ in ws] for k, ws in l3_words_topics.items()}
dump_yaml(l3_words, pl_path3 / "word_list.yml")

agrifood_k_cluster = int(input("Enter the cluster number: "))

l3_topics_all = ex3.force_outlier_assignment(docs, l3_topics, probs, threshold=.4, cluster_index=agrifood_k_cluster)

ex2._plot_path = pl_path3
agrifood_papers = [(1 if t == agrifood_k_cluster else 0) for t in l3_topics_all]
ex2.plot_wonders(docs, add_doc_classes=agrifood_papers, use_training_embeddings=True)

fig_topics = ex3._topic_model.visualize_topics(width=1200, height=1200)
fig_topics.write_html(ex3._plot_path / "topic_space.html")

# --------------------- END PASS 3

save_csv_results(docs, themes=l1_topics, theme_keywords=l1_words, subjects=l2_topics, alt_subjects=l2_topics_all,
                 subj_keywords=l2_words, path=pl_path1.parent / "results", agrifood_papers=agrifood_papers)

# ex.see_topic_evolution(docs, bins_n=3)
