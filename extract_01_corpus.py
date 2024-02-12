import pandas as pd

from topic_extraction.extraction import document_extraction

if __name__ == "__main__":
    ds: pd.Series = pd.read_csv("data/_out_.csv", usecols=["index", "alt_subjects"], index_col="index", dtype={"index": str})["alt_subjects"]

    docs = document_extraction(["t", "a", "k"])

    for c in [0, 1]:
        for d in docs:
            if d.title == "Digital transformation as a springboard for product, process and business model innovation":
                pass
        c_docs = [(f"{d.title.strip()}. {'; '.join([k.strip() for k in d.keywords]) + '. ' if [k for k in d.keywords if k.strip()] else ''}"
                   f"{d.abstract.strip() if d.abstract.strip() != '[No abstract available]' else ''}".strip()) for d in docs if d.id in ds[ds == c].index]
        with open(f"dumps/corpus_{c}.txt", "w") as f:
            f.write("\n".join(c_docs))
