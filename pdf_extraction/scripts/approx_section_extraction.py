import json
from pathlib import Path

from pdfminer.high_level import extract_text
from tqdm import tqdm

from pdf_extraction.classes.pdftree import PDFTree


def main():
    for i in tqdm(range(8239, 9882)):
        tree = PDFTree(i)
        path = Path("data/papers") / str(i)
        text = extract_text(path / tree.json_info['original_name'])
        data = {
            "Abstract": tree.json_info["ris"]["abstract"],
            "Full Text": text
        }
        with open(path / f"content.json", 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
