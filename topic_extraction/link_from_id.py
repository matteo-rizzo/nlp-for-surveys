from __future__ import annotations

from pathlib import Path

import pandas as pd


def get_scopus_link(sid: str) -> str:
    return f"https://www.scopus.com/record/display.uri?eid=2-s2.0-{sid}&origin=inward"


def make_link_set_from_csv(csv_path: str | Path) -> None:
    result_df = pd.read_csv(csv_path, index_col=0, dtype={"paper": str})

    agri_col = result_df["agrifood_cluster"]
    print(agri_col.astype(int).sum())

    result_df["URL"] = result_df.index.map(get_scopus_link)
    result_df.where(result_df["agrifood_cluster"] == 1).dropna()["URL"].to_csv("plots/result_to_send/agri_links.csv")
    result_df.to_csv("plots/result_to_send/link_agrifood.csv")

    # Open all in Web Browser
    # import webbrowser
    # b = webbrowser.get('firefox')
    # for url in link_set:
    #     b.open(url)


if __name__ == "__main__":
    make_link_set_from_csv("plots/result_to_send/NLP_clustering_results - agrifood.csv")
