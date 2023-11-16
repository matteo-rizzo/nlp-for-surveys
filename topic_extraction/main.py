from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
import torch.cuda

from topic_extraction.classes.BERTopicExtractor import BERTopicExtractor
from topic_extraction.classes.Document import Document
from topic_extraction.extraction import document_extraction
from topic_extraction.utils import dump_yaml, save_csv_results, vector_rejection

pd.set_option("display.max_columns", None)

PASS_1 = True
PASS_2 = True

USE_PASS_1_EMBEDDINGS = True  # use embeddings from guided topic modeling from the first model
ORTHOGONAL_SUBJECTS = True  # remove topics from the first model
NORMALIZE_INPUT_EMBEDDINGS = False  # L2-normalization of sentence embeddings
TEXT_COMPOSITION = ["t", "a", "k"]


def list_paper_per_cluster(documents: list[Document], topics: list[int] | np.ndarray[int]) -> dict[int, list[str]]:
    if isinstance(topics, np.ndarray):
        topics: list[int] = topics.tolist()

    document_ids: list[str] = [d.id for d in documents]

    doc_by_cluster: list[tuple[str, int]] = sorted(list(zip(document_ids, topics)), key=lambda x: x[1])

    grouped_docs = dict()  # defaultdict(list)
    for k, g in groupby(doc_by_cluster, key=lambda x: x[1]):
        grouped_docs[k] = [doc_id for doc_id, _ in g]
    return grouped_docs


def get_word_relative_importance(words_topics: dict[int, list[tuple[str, float]]]) -> dict[int, list[tuple[str, float]]]:
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


docs = document_extraction(TEXT_COMPOSITION)

# Pass 1

embeddings = None
theme_embeddings = None
l1_topics = None
if PASS_1:
    # seed_topic_list = [
    #     ["green", "sustainability", "environment", "recycling", "energy", "transition", "transform", "business", "strategy", "model"],  # added strategy, model, automation
    #     ["digitalization", "digital", "technology", "transition", "transform", "business", "strategy", "model"],
    # ]

    # SEED documents (notes)
    # 0 [8, 169, 274, 481, 496, 557, 909, 1139, 1270, 1358, 1474, 1504, 1533, 1580]
    # ['85147412525', '85139910222', '85132285640', '85133459108', '85131347722', '85120406970', '85100997792', '85074162493', '85074398063', '85054261475', '85030749889', '85043590608', '85038637535', '85030468164']

    # 1 [288, 586, 720, 726, 939, 1422, 1506, 1829, 1858]
    # ['85119652020', '85120786316', '85101084432', '85096444313', '85070896886', '85087158749', '85042455591', '84857671100', '79957520956']

    supervised_labels: pd.Series = pd.read_csv("data/supervised_sample.csv", index_col="index", dtype={"index": str})["0"]
    y = [supervised_labels[doc.id] if doc.id in supervised_labels.index else -1 for doc in docs]
    # y = None

    # --------------------- PASS 1
    pl_path1 = Path("plots") / "themes"
    pl_path1.mkdir(exist_ok=True, parents=True)
    ex1 = BERTopicExtractor(plot_path=pl_path1)
    ex1.prepare(config_file="topic_extraction/config/bertopic1.yml", seed_topic_list=None)

    if Path(ex1._embedding_save_path).is_file():
        embeddings = np.load(ex1._embedding_save_path)

    # Documents representative of guided TM
    # for i, d in enumerate(docs):
    #     if i in [288, 586, 720, 726, 939, 1422, 1506, 1829, 1858]:  # [8, 169, 274, 481, 496, 557, 909, 1139, 1270, 1358, 1474, 1504, 1533, 1580]:
    #         print(d.title)

    l1_topics, l1_probs, l1_raw_probs, l1_words_topics = ex1.train(docs, embeddings=embeddings, normalize=NORMALIZE_INPUT_EMBEDDINGS, y=y)

    # l1_topics, l1_probs, l1_raw_probs, l1_words_topics = ex1.batch_extract(docs, -1, use_training_embeddings=True)
    print(f"L1 outliers pre-reduction: {len([t for t in l1_topics if t < 0])}")
    l1_topics = ex1._topic_model.reduce_outliers([d.body for d in docs], l1_topics, probabilities=l1_raw_probs, strategy="probabilities", threshold=.3)
    print(f"L1 outliers post-reduction: {len([t for t in l1_topics if t < 0])}")

    # Save probabilities for testing
    multilabel_probs = pd.DataFrame(l1_raw_probs, index=[d.id for d in docs])
    multilabel_probs.to_csv("plots/l1_probs_results_hdbscan.csv", index_label="index")

    torch.cuda.empty_cache()
    theme_embeddings = ex1._topic_model.topic_embeddings_[1:]
    embeddings = ex1._train_embeddings

    # Save them for usage in tuning-routine
    np.save("dumps/embeddings/gtm_embeddings.npy", embeddings)
    np.save("dumps/embeddings/theme_embeddings.npy", theme_embeddings)

    del ex1

    # Plot/save results
    # ex1.plot_wonders(docs)

    l1_words = get_word_relative_importance(l1_words_topics)
    dump_yaml(l1_words, pl_path1 / "word_list.yml")

# --------------------- END PASS 1


if PASS_2:
    # --------------------- PASS 2
    # Determine field of application

    file_suffix = "".join(TEXT_COMPOSITION)

    pl_path2 = Path("plots") / "fields"
    pl_path2.mkdir(exist_ok=True, parents=True)
    ex2 = BERTopicExtractor(plot_path=pl_path2)
    ex2.prepare(config_file=f"topic_extraction/config/bertopic2_{file_suffix}.yml")

    if not (USE_PASS_1_EMBEDDINGS and PASS_1):
        # Compute embeddings from scratch
        embeddings = None
        if Path(ex2._embedding_save_path).is_file():
            # Reload embeddings
            embeddings = np.load(ex2._embedding_save_path)

    if ORTHOGONAL_SUBJECTS and PASS_1:
        # Project embeddings in other space to remove unwanted themes
        embeddings = vector_rejection(embeddings, theme_embeddings)

    l2_topics, l2_probs, l2_raw_probs, l2_words_topics = ex2.train(docs, normalize=NORMALIZE_INPUT_EMBEDDINGS, embeddings=embeddings, reduce_outliers=True, threshold=.5)
    print(f"DBCV: {ex2._topic_model.hdbscan_model.relative_validity_}")
    # l2_topics, l2_probs, l2_raw_probs, l2_words_topics = ex2.batch_extract(docs, -1, use_training_embeddings=True, reduce_outliers=True, threshold=.5)
    print(f"Found {max(l2_topics) + 1} subjects.")

    grouped_papers = list_paper_per_cluster(docs, l2_topics)

    ex2.plot_wonders(docs, add_doc_classes=l1_topics, use_training_embeddings=True, file_suffix=file_suffix)
    l2_topics_all_prob = ex2._topic_model.reduce_outliers([d.body for d in docs], l2_topics, probabilities=l2_raw_probs, strategy="probabilities", threshold=.3)
    # l2_topics_all_dist = ex2._topic_model.reduce_outliers([d.body for d in docs], l2_topics, strategy="distributions", threshold=.3)
    # l2_topics_all = [p if p == d else -1 for p, d in zip(l2_topics_all_prob, l2_topics_all_dist)]
    l2_topics_all = l2_topics_all_prob
    print(f"Outliers with forced subjects: {len([t for t in l2_topics_all if t < 0])}")

    l2_words = get_word_relative_importance(l2_words_topics)
    dump_yaml(l2_words, pl_path2 / f"word_list_{file_suffix}.yml")

    torch.cuda.empty_cache()
    # --------------------- END PASS 2

# --------------------- PASS 3

# seed_topic_list2 = [
#     ['fish', 'harvest', 'agro-food', 'agri-food', 'agrotourism', 'agro-chemical', 'horticulture', 'agriculture', 'agroecology', 'husbandry', 'agrifood', 'agribusiness',
#      'agrochemical', 'farmer', 'bier', 'agro-industry', 'agroforestry', 'farm', 'farmland', 'aquaculture', 'crop growing', 'farmwork', 'agri-business', 'agroindustry',
#      'sharecropping', 'agricultural', 'wine', 'cultivation', 'viticulture', 'beer', 'hydroponics', 'agrofood', 'food', 'farming', 'agronomy', 'livestock', 'agritourism',
#      'agrifood-tech',
#      "agri-food system", "agri-food ecosystem", "agri-food firm", "food system", "bio-district", "digital transformation in agriculture", "food value chain",
#      "sustainable agriculture", "forest"]
# ]

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
    save_csv_results(docs, themes=l1_topics, subjects=l2_topics, alt_subjects=l2_topics_all,
                     subj_keywords=l2_words, theme_keywords=l1_words,
                     csv_path=pl_path1.parent / "results",
                     papers_by_subject=grouped_papers,
                     agrifood_papers=None, theme_probs=l1_raw_probs, subj_probs=l2_probs, write_ods=True,
                     file_suffix=file_suffix)

# ex.see_topic_evolution(docs, bins_n=3)
