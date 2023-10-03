from pathlib import Path

import numpy as np
import pandas as pd
import torch.cuda

from topic_extraction.classes.BERTopicExtractor import BERTopicExtractor
from topic_extraction.extraction import document_extraction
from topic_extraction.utils import dump_yaml, save_csv_results

pd.set_option("display.max_columns", None)

PASS_1 = False
PASS_2 = True


def remove_unwanted_cluster(doc_embeddings, unwanted_embeddings):
    # Compute centroid of unwanted cluster
    u = np.mean(unwanted_embeddings, axis=0)

    # Subtract projection onto the unwanted direction
    new_embeddings = doc_embeddings - np.outer(np.dot(doc_embeddings, u) / np.dot(u, u), u)

    return new_embeddings


def get_word_relative_importance(words_topics: dict[str, list[tuple[str, float]]]) -> dict[str, list[tuple[str, float]]]:
    """
    Weight the importance of representative keywords

    :param words_topics: topic keywords for each cluster
    :return: dictionary as the input but with weighted importance
    """

    # sum_importance = {k: sum([s for _, s in ws]) for k, ws in words_topics.items()}

    # words = {k: [(w, float(s / sum_importance[k])) for w, s in ws] for k, ws in words_topics.items()}
    words = {k: [(w, float(s)) for w, s in ws] for k, ws in words_topics.items()}
    # words_score = {k: [s / sum_importance[k] for _, s in ws] for k, ws in words_topics.items()}
    return words


docs = document_extraction()

# Pass 1

l1_topics = None
if PASS_1:
    # seed_topic_list = [
    #     ["circular economy", "sustainability", "sustainable business", "sustainable development", "recycling", "waste", "green transition", "green transform"],
    #     ["digitalization", "digital business", "digital economy", "digital innovation", "business transformation"]
    # ]

    # seed_topic_list1 = [
    #     ["green", "sustainability", "environment", "green transition", "green transform", "sustainable transformation", "sustainable business"],
    #     ["digitalization", "digital", "digital transition", "digital transformation", "e-business", "digital business", "automation"],
    # ]

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

    ex1.train(docs, embeddings=embeddings, normalize=False)

    l1_topics, l1_probs, l1_words_topics = ex1.batch_extract(docs, -1, use_training_embeddings=True)
    torch.cuda.empty_cache()
    theme_embeddings = ex1._topic_model.topic_embeddings_
    # embeddings = ex1._train_embeddings
    del ex1

    # Plot/save results
    # ex1.plot_wonders(docs)

    l1_words = get_word_relative_importance(l1_words_topics)
    dump_yaml(l1_words, pl_path1 / "word_list.yml")

# --------------------- END PASS 1


if PASS_2:
    # --------------------- PASS 2
    # Determine field of application

    pl_path2 = Path("plots") / "fields"
    pl_path2.mkdir(exist_ok=True, parents=True)
    ex2 = BERTopicExtractor(plot_path=pl_path2)
    ex2.prepare(config_file="topic_extraction/config/bertopic2.yml")

    embeddings = None
    if Path(ex2._embedding_save_path).is_file():
        embeddings = np.load(ex2._embedding_save_path)
        # Project embeddings in other space to remove unwanted themes
        # embeddings = remove_unwanted_cluster(embeddings, theme_embeddings[1:])

    ex2.train(docs, normalize=False, embeddings=embeddings)
    print(f"DBCV: {ex2._topic_model.hdbscan_model.relative_validity_}")
    l2_topics, l2_probs, l2_words_topics = ex2.batch_extract(docs, -1, use_training_embeddings=True, reduce_outliers=True, threshold=.5)
    ex2.plot_wonders(docs, add_doc_classes=l1_topics, use_training_embeddings=True)
    # l2_topics_all = ex2._topic_model.reduce_outliers([d.body for d in docs], l2_topics, probabilities=probs, strategy="probabilities", threshold=.3)

    l2_words = get_word_relative_importance(l2_words_topics)
    dump_yaml(l2_words, pl_path2 / "word_list.yml")

    torch.cuda.empty_cache()
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

# seed_topic_list2 = [
#     ["agrifood", "agri", "agronomy", "nutrition", "ecological", "cultivation", "farm", "agriculture", "crops", "aquaculture", "agroecology", "crop growing",
#      "livestock", "food", "viticulture", "wine", "beer", "bier", "farmland", "harvest", "agri-food industry", "sharecropping", "agroindustry", "agroforestry",
#      "agro-tourism", "hydroponics", "farmwork", "husbandry", "horticulture", "fish", "agriculture technology",
#      "sustainability-driven nutrient roadmap", "nutritious crops", "nutrient recovery", "climate-smart fertilizers", "digital crop nutrition",
#      "seed systems", "farmer", "agri-food system", "agri-food ecosystem", "agri-food firm",
#      "food system", "bio-district", "transition agri-food sustainability", "agribusiness", "e-agribusiness", "agribusiness corporation",
#      "agrifood‐tech", "agrifood‐tech e‐business", "agribusiness sector", "digital transformation in agriculture", "food value chain", "food hub",
#      "sustainability", "sustainability of agriculture products", "food security sustainability", "countries local food", "sustainable agriculture",
#      "sustainable development", "agricultural sector", "sustainable regional development", "forest bioeconomy", "ecosystem innovation", "organic farming"]
# ]

# *** DISABLED agrifood for now
# pl_path3 = Path("plots") / "agrifood"
# pl_path3.mkdir(exist_ok=True, parents=True)
# ex3 = BERTopicExtractor(plot_path=pl_path3)
# ex3.prepare(config_file="topic_extraction/config/bertopic3.yml", seed_topic_list=seed_topic_list2)  # dimensionality_reduction=ex2._reduction_model
# del ex2._embedding_model
# torch.cuda.empty_cache()
#
# embeddings = None
# if Path(ex3._embedding_save_path).is_file():
#     embeddings = np.load(ex3._embedding_save_path)
#
# ex3.train(docs, normalize=True, embeddings=embeddings, fit_reduction=True)
# print(f"DBCV: {ex3._topic_model.hdbscan_model.relative_validity_}")
# l3_topics, probs, l3_words_topics = ex3.batch_extract(docs, -1, use_training_embeddings=True)
# l3_words = {k: [w for w, _ in ws] for k, ws in l3_words_topics.items()}
# dump_yaml(l3_words, pl_path3 / "word_list.yml")
#
# agrifood_k_cluster = int(input("Enter the cluster number: "))
#
# l3_topics_all = ex3.force_outlier_assignment(docs, l3_topics, probs, threshold=.4, cluster_index=agrifood_k_cluster)
#
# ex2._plot_path = pl_path3
# agrifood_papers = [(1 if t == agrifood_k_cluster else 0) for t in l3_topics_all]
# ex2.plot_wonders(docs, add_doc_classes=agrifood_papers, use_training_embeddings=True)
#
# fig_topics = ex3._topic_model.visualize_topics(width=1200, height=1200)
# fig_topics.write_html(ex3._plot_path / "topic_space.html")

# --------------------- END PASS 3

if PASS_2 and PASS_1:
    save_csv_results(docs, themes=l1_topics, theme_keywords=l1_words, subjects=l2_topics, alt_subjects=None,
                     subj_keywords=l2_words, csv_path=pl_path1.parent / "results", agrifood_papers=None, theme_probs=l1_probs, subj_probs=l2_probs)

# ex.see_topic_evolution(docs, bins_n=3)
