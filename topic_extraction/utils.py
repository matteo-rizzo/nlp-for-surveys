from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from topic_extraction.classes.Document import Document


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


def save_csv_results(docs: list[Document], themes: list[int], subjects: list[int], alt_subjects: list[int],
                     theme_keywords: dict[list[str]], subj_keywords: dict[list[str]], path: str | Path, agrifood_papers: list[int] = None) -> None:
    """
    Save clustering results to CSV file

    :param docs: documents
    :param themes: 1st level of labels
    :param subjects: 2nd level of labels
    :param alt_subjects: 2nd level of labels fixed
    :param theme_keywords: keywords associated with 1st level topics
    :param subj_keywords: keywords associated with 2nd level topics
    :param path:
    :return:
    """

    path.mkdir(exist_ok=True, parents=True)

    ids = [str(d.id) for d in docs]

    tk = pd.DataFrame(theme_keywords).to_csv(path / "themes.csv")
    sk = pd.DataFrame(subj_keywords).to_csv(path / "subjects.csv")

    a_args = dict()
    if agrifood_papers:
        a_args["agrifood"] = agrifood_papers
    if alt_subjects:
        a_args["alt_subjects"] = alt_subjects
    classification_df = pd.DataFrame(dict(themes=themes, subjects=subjects, **a_args), index=ids)
    classification_df.to_csv(path / "classification.csv")
