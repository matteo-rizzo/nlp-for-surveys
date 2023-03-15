import pandas as pd


def metadata_extraction():
    df = pd.read_csv("data/TwinTransitionEstrazione7mar2023.csv")

    print(df[['Authors', 'Author(s) ID']].head())
    print(df.columns)


def text_extraction() -> tuple[list[str], list[str]]:
    df = pd.read_csv("data/TwinTransitionEstrazione7mar2023.csv", usecols=["Title", "Abstract"])
    df["all"] = df["Title"].str.cat(df["Abstract"], sep=". ")
    return df["all"].tolist(), df["Title"].tolist()


if __name__ == "__main__":
    metadata_extraction()
