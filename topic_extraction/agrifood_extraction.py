from pathlib import Path

import numpy as np
import pandas as pd

from topic_extraction.classes.BERTopicExtractor import BERTopicExtractor
from topic_extraction.extraction import document_extraction
import re

pd.set_option("display.max_columns", None)

docs = document_extraction()

if __name__ == "__main__":

    # pattern = r"agri|agro"
    # matching = set([d.id for d in docs if re.search(pattern, d.title)])
    # matching |= set([d.id for d in docs if re.search(pattern, d.body)])
    # matching |= set([d.id for d in docs if any([re.search(pattern, k) for k in d.keywords])])

    with open("agrifood.txt", mode="r") as f:
        matching = set(f.readlines())

    supervised_list = [(1 if d.id in matching else -1) for d in docs]

    seed_topic_list = None # [
    #     ['fish', 'harvest', 'agro-food', 'agri-food', 'agrotourism', 'agro-chemical', 'horticulture', 'agriculture', 'agroecology', 'husbandry', 'agrifood', 'agribusiness',
    #      'agrochemical', 'farmer', 'bier', 'agro-industry', 'agroforestry', 'farm', 'farmland', 'aquaculture', 'crop growing', 'farmwork', 'agri-business', 'agroindustry',
    #      'sharecropping', 'agricultural', 'wine', 'cultivation', 'viticulture', 'beer', 'hydroponics', 'agrofood', 'food', 'farming', 'agronomy', 'livestock', 'agritourism',
    #      'agrifood-tech',
    #      "agri-food system", "agri-food ecosystem", "agri-food firm", "food system", "bio-district", "digital transformation in agriculture", "food value chain",
    #      "sustainable agriculture", "forest"]
    #     # ["agrifood", "agri", "agronomy", "nutrition", "ecological", "cultivation", "farm", "agriculture", "crops", "aquaculture", "agroecology", "crop growing",
    #     #  "livestock", "food", "viticulture", "wine", "beer", "bier", "farmland", "harvest", "agri-food industry", "sharecropping", "agroindustry", "agroforestry",
    #     #  "agro-tourism", "hydroponics", "farmwork", "husbandry", "horticulture", "fish", "agriculture technology",
    #     #  "sustainability-driven nutrient roadmap", "nutritious crops", "nutrient recovery", "climate-smart fertilizers", "digital crop nutrition",
    #     #  "seed systems", "climate adaptation", "farmer", "agri-food system", "agri-food ecosystem", "agri-food firm",
    #     #  "food system", "bio-district", "organic", "transition agri-food sustainability", "agribusiness", "e-agribusiness", "agribusiness corporation",
    #     #  "agrifood‐tech", "agrifood‐tech e‐business", "agribusiness sector", "BMC framework", "digital transformation in agriculture", "food value chain", "food hub",
    #     #  "sustainability", "sustainability of agriculture products", "food security sustainability", "countries local food", "sustainable agriculture",
    #     #  "sustainable development", "agricultural sector", "regional development", "forest bioeconomy", "ecosystem innovation", "organic farming"]
    # ]

    pl_path3 = Path("plots") / "agrifood"
    pl_path3.mkdir(exist_ok=True, parents=True)
    ex3 = BERTopicExtractor(plot_path=pl_path3)
    ex3.prepare(config_file="topic_extraction/config/bertopic3.yml", seed_topic_list=seed_topic_list)

    embeddings = None
    if Path(ex3._embedding_save_path).is_file():
        embeddings = np.load(ex3._embedding_save_path)
    # torch.cuda.empty_cache()

    ex3.train(docs, embeddings=embeddings, y=supervised_list)
    print(f"DBCV: {ex3._topic_model.hdbscan_model.relative_validity_}")
    l3_topics, probs, l3_words_topics = ex3.batch_extract(docs, -1, use_training_embeddings=True)
    ex3.plot_wonders(docs, add_doc_classes=[(1 if d.id in matching else 0) for d in docs], use_training_embeddings=True)

    ids = [str(d.id) for d in docs]
    pd.DataFrame(dict(cluster=l3_topics), index=ids).to_csv(pl_path3 / "classification.csv")
