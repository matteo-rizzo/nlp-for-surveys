from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from topic_extraction.classes.Document import Document


def expand_scores(keywords_dict: dict) -> dict:
    """ Utility to pretty insert columns in dataframe """
    words_score = {f"{k}_scores": [s for _, s in ws] for k, ws in keywords_dict.items()}
    words_with_score = {
        **{k: [w for w, _ in ws] for k, ws in keywords_dict.items()},
        **words_score
    }
    return words_with_score


def load_yaml(path: str | Path) -> Any:
    """
    Load YAML as python dict

    @param path: path to YAML file
    @return: dictionary containing data
    """
    with open(path, encoding="UTF-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def dump_yaml(data, path: str | Path) -> None:
    """
    Load YAML as python dict

    @param path: path to YAML file
    @param data: data to dump
    @return: dictionary containing data
    """
    with open(path, encoding="UTF-8", mode="w") as f:
        yaml.dump(data, f, Dumper=yaml.SafeDumper)


def save_csv_results(docs: list[Document], themes: list[int], subjects: list[int], alt_subjects: list[int] | None,
                     theme_keywords: dict[list[str]], subj_keywords: dict[list[str]], csv_path: str | Path,
                     agrifood_papers: list[int] = None, theme_probs: list[float] | None = None, subj_probs: list[float] | None = None) -> None:
    """
    Save clustering results to CSV file

    :param docs: documents
    :param themes: 1st level of labels
    :param subjects: 2nd level of labels
    :param alt_subjects: 2nd level of labels fixed
    :param theme_keywords: keywords associated with 1st level topics
    :param subj_keywords: keywords associated with 2nd level topics
    :param agrifood_papers: agrifood topics from clustering
    :param csv_path: the path where to write results
    :param subj_probs: confidence for cluster assignment of subjects
    :param theme_probs: confidence for cluster assignment with themes
    """

    csv_path.mkdir(exist_ok=True, parents=True)

    ids = [str(d.id) for d in docs]

    assert theme_probs is None or len(theme_probs) == len(themes), f"Themes probabilities and assigned clusters have different sizes: {len(theme_probs)} - {len(themes)}"
    assert subj_probs is None or len(subj_probs) == len(subjects), f"Subjects probabilities and assigned clusters have different sizes: {len(subj_probs)} - {len(subjects)}"

    pd.DataFrame(expand_scores(theme_keywords)).to_csv(csv_path / "themes.csv")
    pd.DataFrame(expand_scores(subj_keywords)).to_csv(csv_path / "subjects.csv")

    a_args = dict()
    if agrifood_papers:
        a_args["agrifood"] = agrifood_papers
    if alt_subjects:
        a_args["alt_subjects"] = alt_subjects
    if theme_probs is not None:
        a_args["themes_prob"] = theme_probs
    if subj_probs is not None:
        a_args["subj_prob"] = subj_probs
    classification_df = pd.DataFrame(dict(themes=themes, subjects=subjects, **a_args), index=ids)
    classification_df.to_csv(csv_path / "classification.csv")
