from pathlib import Path

import torch.cuda

from topic_extraction.classes.BERTopicExtractor import BERTopicExtractor
from topic_extraction.extraction import document_extraction

# K-MEANS best (specter): 15
# HDBSCAN best (specter): 'min_samples': 3, 'min_cluster_size': 15, 'metric': 'euclidean', 'cluster_selection_method': 'eom'

docs = document_extraction()

pl_path1 = Path("plots") / "validation"
ex1 = BERTopicExtractor(plot_path=pl_path1)
ex1.prepare(config_file="topic_extraction/config/bertopic.yml")
# ex1.find_optimal_n_clusters(docs, conf_search="topic_extraction/config/model_selection.yml", normalize=False, result_path="allenai_hdbscan_grid_results")
ex1.find_optimal_n_clusters(docs, conf_search="topic_extraction/config/model_selection.yml", normalize=True, result_path="allenai-specter_hdbscan_grid_results_norm")

torch.cuda.empty_cache()

