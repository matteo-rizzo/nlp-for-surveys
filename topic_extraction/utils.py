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


def save_csv_results(docs: list[Document], themes: list[int], subjects: list[int], theme_keywords: dict[list[str]], subj_keywords: dict[list[str]], path: str | Path) -> None:
    """
    Data

    :param data:
    :param path:
    :return:
    """

    path.mkdir(exist_ok=True, parents=True)

    ids = [str(d.id) for d in docs]

    tk = pd.DataFrame(theme_keywords).to_csv(path / "themes.csv")
    sk = pd.DataFrame(subj_keywords).to_csv(path / "subjects.csv")

    classification_df = pd.DataFrame(dict(themes=themes, subjects=subjects), index=ids).to_csv(path / "classification.csv")
