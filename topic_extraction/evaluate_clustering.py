from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

TARGET_FILE = "plots/results/shared_results/title_abstract_keywords/all_results_tak.ods"
BENCHMARK_FILE = "data/DIG and GREEN papers.xlsx"


def extract_id(a: pd.Series):
    sids: list[str | None] = list()
    for link in a.values.tolist():
        sid = re.findall("eid=2-s2.0-(\\d+)", link)
        if sid:
            sids.append(sid[0])
        else:
            sids.append(None)
    return sids


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray, sk_classifier_name: str = None) -> dict:
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=0)
    acc = accuracy_score(y_true, y_pred)
    if sk_classifier_name:
        print(f"{sk_classifier_name} accuracy: {acc:.3f}")
        print(f"{sk_classifier_name} precision: {precision:.3f}")
        print(f"{sk_classifier_name} recall: {recall:.3f}")
        print(f"{sk_classifier_name} F1-score: {f1_score:.3f}")

    return {"f1": f1_score, "accuracy": acc, "precision": precision, "recall": recall}


def extract_supervised_sample():
    df_truth: dict[str, pd.DataFrame] = pd.read_excel(BENCHMARK_FILE, sheet_name=None, header=None, names=["link"])
    df_truth: pd.Series = pd.Series(dict([a for b in (list(zip(extract_id(df["link"]), [i] * df.shape[0])) for i, df in enumerate(df_truth.values())) for a in b]))
    df_truth.name = "target"

    additional_1 = ['85147412525', '85139910222', '85132285640', '85133459108', '85131347722', '85120406970', '85100997792', '85074162493', '85074398063', '85054261475',
                    '85030749889', '85043590608', '85038637535', '85030468164']
    additional_0 = ['85119652020', '85120786316', '85101084432', '85070896886', '85087158749', '85042455591', '84857671100', '79957520956']
    additional_truth = pd.Series(
        index=additional_1 + additional_0, data=([1] * len(additional_1)) + ([0] * len(additional_0)))
    additional_truth = pd.concat([df_truth, additional_truth])

    additional_truth.to_csv("data/supervised_sample.csv", index_label="index")


if __name__ == "__main__":
    df_target: dict[str, pd.DataFrame] = pd.read_excel(BENCHMARK_FILE, sheet_name=None, header=None, names=["link"])

    # TODO: implement, depending on format
    df_pred: pd.Series = None
    df_pred.name = "prediction"

    df_target: pd.Series = pd.Series(dict([a for b in (list(zip(extract_id(df["link"]), [i] * df.shape[0])) for i, df in enumerate(df_target.values())) for a in b]))
    df_target.name = "target"

    all_data = pd.concat([df_pred, df_target], join="inner", axis=1, verify_integrity=True)

    compute_metrics(all_data["prediction"].tolist(), all_data["target"].tolist(), "Clustering KMeans")
