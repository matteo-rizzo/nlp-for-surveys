from __future__ import annotations

import re

import pandas as pd

from topic_extraction.classes.Document import Document


def metadata_extraction():
    df = pd.read_csv("data/TwinTransitionEstrazione7mar2023.csv")

    print(df[['Authors', 'Author(s) ID']].head())
    print(df.columns)

    years: list[str] = [y.strip() for y in df["Year"].astype(str).tolist()]
    non_years = [y for y in years if not (isinstance(y, str) and len(y) == 4)]
    print(non_years)

    non_keywords: list[str] = [s for s in df["Author Keywords"].tolist() if not (isinstance(s, str) and len(s) > 0)]
    print(non_keywords)

    years: list[int] = df["Year"].astype(int).tolist()
    print(f"Min/max year: {min(years), max(years)}")


def text_extraction() -> tuple[list[str], list[str], list[str], list[list[str]]]:
    df = pd.read_csv("data/TwinTransitionEstrazione7mar2023.csv", usecols=["Title", "Abstract", "Year", "Author Keywords"])
    df["all"] = df["Title"].str.cat(df["Abstract"], sep=". ")

    keywords: list[list[str]] = [[c.strip() for c in s.split(";")] for s in df["Author Keywords"].fillna("").tolist()]
    years: list[str] = [y.strip() for y in df["Year"].astype(str).tolist()]
    return df["all"].tolist(), df["Title"].tolist(), years, keywords


def extract_scopus_id(eid: str) -> str:
    sid = re.findall("^2-s2.0-(\\d+)$", eid)
    s = ""
    if sid:
        s = sid[0]
    return s


def document_extraction() -> list[Document]:
    df = pd.read_csv("data/TwinTransitionEstrazione7mar2023.csv", usecols=["Title", "Abstract", "Year", "Author Keywords", "EID"])
    df["all"] = df["Title"].str.cat(df["Abstract"], sep=". ")

    # TODO: add extraction of missing Scopus IDs from URLs

    keywords: list[list[str]] = [[c.strip() for c in s.split(";")] for s in df["Author Keywords"].fillna("").tolist()]
    years: list[int] = [y for y in df["Year"].astype(int).tolist()]
    ids: list[str] = [extract_scopus_id(s) for s in df["EID"].fillna("").tolist()]
    docs = [Document(i, b, t, k, y) for i, b, t, y, k in zip(ids, df["all"].tolist(), df["Title"].tolist(), years, keywords)]
    return docs


if __name__ == "__main__":
    metadata_extraction()
