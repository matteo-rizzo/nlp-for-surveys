from __future__ import annotations

import re
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# TARGET_FILE = "plots/results/shared_results/title_abstract_keywords/all_results_tak.ods"
TARGET_FILE = "plots/results/all_results_tak.ods"
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
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
    acc = accuracy_score(y_true, y_pred)

    # Find metrics per combination
    # y_pred_df = pd.DataFrame(y_pred)
    # y_true_df = pd.DataFrame(y_true)
    # y_pred_df["cats"] = y_pred_df.iloc[:, 0].astype(str).str.cat(y_pred_df.iloc[:, 1].astype(str), sep="")
    # y_true_df["cats"] = y_true_df.iloc[:, 0].astype(str).str.cat(y_true_df.iloc[:, 1].astype(str), sep="")
    # y_pred_df = y_pred_df[y_pred_df["cats"] == "00"]["cats"]
    # y_pred_df.name = "pred"
    # y_true_df = y_true_df[y_true_df["cats"] == "00"]["cats"]
    # y_true_df.name = "target"
    # d = pd.concat([y_true_df, y_pred_df], axis=1, join="outer").fillna("11")
    # pprint(precision_recall_fscore_support(d["target"], d["pred"], average="binary", pos_label="00"))

    if sk_classifier_name:
        print(f"{sk_classifier_name} accuracy: {acc:.3f}")
        print(f"{sk_classifier_name} precision: {precision:.3f}")
        print(f"{sk_classifier_name} recall: {recall:.3f}")
        print(f"{sk_classifier_name} F1-score: {f1_score:.3f}")
        print(f"{sk_classifier_name} FLAT accuracy: {accuracy_score(y_true.reshape(-1), y_pred.reshape(-1)):.3f}")

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


def evaluate_twin_papers(thr: float = .3):
    df_target: pd.DataFrame = pd.read_csv("data/benchmark_data_raw.csv", usecols=["index", "digital", "green"], index_col="index",
                                          dtype={"index": str, "digital": float, "green": float})
    df_target.fillna(.0, inplace=True)
    df_target["digital"] = df_target.apply(lambda row: 1 if row.digital >= 20 else 0, axis=1).astype(bool)
    df_target["green"] = df_target.apply(lambda row: 1 if row.green >= 20 else 0, axis=1).astype(bool)
    df_target["twin"] = (df_target["digital"] & df_target["green"]).astype(int)
    # df_target = df_target.rename(columns={"digital": "1", "green": "0"}).sort_index(axis="columns")
    # df_target = df_target / 100.0
    # df_pred: pd.DataFrame = pd.read_csv("plots/l1_probs_results_hdbscan.csv", index_col="index", usecols=["index", "0", "1"], dtype={"index": str, "0": float, "1": float})
    with pd.ExcelFile("plots/third_results/tak_4/all_results_tak.xlsx") as exc_file:
        df_pred: pd.DataFrame = pd.read_excel(exc_file, sheet_name="classification", index_col="index", usecols=["index", "theme_0_prob", "theme_1_prob"],
                                              dtype={"index": str, "theme_0_prob": float, "theme_1_prob": float})

    df_pred = df_pred.loc[df_pred.index.intersection(df_target.index), :].sort_index(ascending=True)
    df_target = df_target.sort_index(ascending=True)
    df_pred = (df_pred >= thr).astype(bool)
    df_pred["twin"] = (df_pred["theme_0_prob"] & df_pred["theme_1_prob"]).astype(int)
    assert set(df_pred.index.tolist()) == set(df_target.index.tolist()), "Indexes differ!"
    print(np.count_nonzero(df_pred["twin"].to_numpy()))
    print(np.count_nonzero(df_target["twin"].to_numpy()))
    compute_metrics(y_pred=df_pred["twin"].to_numpy(), y_true=df_target["twin"].to_numpy(), sk_classifier_name="HDBSCAN results")


if __name__ == "__main__":
    evaluate_twin_papers(.38)
    exit(0)

    df_target: pd.DataFrame = pd.read_csv("data/benchmark_data_raw.csv", usecols=["index", "digital", "green"], index_col="index",
                                          dtype={"index": str, "digital": float, "green": float})
    df_target.fillna(.0, inplace=True)
    df_target["digital"] = df_target.apply(lambda row: 1 if row.digital > 10 else 0, axis=1).astype(int)
    df_target["green"] = df_target.apply(lambda row: 1 if row.green > 10 else 0, axis=1).astype(int)
    # df_target = df_target.rename(columns={"digital": "1", "green": "0"}).sort_index(axis="columns")
    # df_target = df_target / 100.0
    df_pred: pd.DataFrame = pd.read_csv("plots/l1_probs_results_hdbscan.csv", index_col="index", usecols=["index", "0", "1"], dtype={"index": str, "0": float, "1": float})
    df_pred = df_pred.loc[df_pred.index.intersection(df_target.index), :].sort_index(ascending=True)
    df_target = df_target.sort_index(ascending=True)
    df_pred = (df_pred > 0.3).astype(int)
    assert set(df_pred.index.tolist()) == set(df_target.index.tolist()), "Indexes differ!"
    compute_metrics(y_pred=df_pred.to_numpy(), y_true=df_target.to_numpy(), sk_classifier_name="HDBSCAN results")

    # # df_target = df_target.rename(columns={"digital": "green", "green": "digital"}).sort_index(axis="columns")
    # df_target["label"] = df_target.apply(lambda row: 0 if row.digital >= 60 else -1, axis=1)
    # df_target["label"] = df_target.apply(lambda row: 1 if row.green >= 60 else row.label, axis=1)
    # df_target["label"] = df_target.apply(lambda row: -1 if row.green < 60 and row.digital < 60 else row.label, axis=1)
    # df_target["label"] = df_target["label"].astype(int)
    # df_target = df_target["label"]
    # df_target = df_target[(df_target >= 0) & (df_target <= 1)]
    # df_target.name = "target"
    #
    # df_pred: pd.Series = pd.read_excel(TARGET_FILE, sheet_name="classification", index_col="index", usecols=["index", "themes"], dtype={"index": str, "themes": int})["themes"]
    # df_pred.name = "prediction"
    # df_pred = df_pred[(df_pred >= 0) & (df_pred <= 1)]
    #
    # all_data = pd.concat([df_pred, df_target], join="inner", axis=1, verify_integrity=True)
    #
    # compute_metrics(all_data["prediction"].tolist(), all_data["target"].tolist(), "Clustering KMeans")
