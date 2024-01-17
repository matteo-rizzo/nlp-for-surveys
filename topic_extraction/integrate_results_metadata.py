from collections import defaultdict
from pathlib import Path

import pandas as pd

from topic_extraction.extraction import document_extraction

path = Path("plots") / "third_results" / "tak_4"

if __name__ == "__main__":
    docs = document_extraction(["t", "a", "k"])

    with pd.ExcelFile(path / "all_results_tak.xlsx") as exc_file:
        df_pred: pd.DataFrame = pd.read_excel(exc_file, sheet_name="classification", index_col="index",
                                              dtype={"index": str, "theme_0_prob": float, "theme_1_prob": float})
        theme_df = pd.read_excel(exc_file, sheet_name="themes")
        subjects_df = pd.read_excel(exc_file, sheet_name="subjects")

    metadata = defaultdict(list)

    for d in docs:
        metadata["authors"].append(d.authors)
        metadata["year"].append(d.timestamp)
        metadata["keywords"].append("; ".join(d.keywords) if d.keywords else "")
        metadata["title"].append(d.title)
        metadata["abstract"].append(d.abstract)
        metadata["source title"].append(d.source)
        metadata["document type"].append(d.doc_type)

    df_meta = pd.DataFrame(metadata, index=pd.Index([d.id for d in docs], dtype=str, name="index"))

    df_pred_ext = pd.concat([df_pred, df_meta], join="inner", axis=1)

    with pd.ExcelWriter(path / "all_results_tak_ext.ods", engine="odf") as exc_writer:
        df_pred_ext.to_excel(exc_writer, sheet_name="classification", index=True)
        theme_df.to_excel(exc_writer, sheet_name="themes", index=False)
        subjects_df.to_excel(exc_writer, sheet_name="subjects", index=False)
