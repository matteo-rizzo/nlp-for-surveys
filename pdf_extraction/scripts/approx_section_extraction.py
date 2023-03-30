import json
from pathlib import Path

from pdfminer.high_level import extract_text
from tqdm import tqdm

from pdf_extraction.classes.pdftree import PDFTree


def abstract_and_text_to_json():
    for i in tqdm(range(8239, 9882)):
        tree = PDFTree(i)
        path = Path("data/papers") / str(i)
        text = extract_text(path / tree.json_info['original_name'])
        data = {
            "Abstract": tree.json_info["ris"].get("abstract", ""),
            "Full Text": text
        }
        with open(path / f"content.json", 'w') as f:
            json.dump(data, f, indent=2)


def clean_text():
    # Ok it crashed after a while so only 178 were done but we'll fix that later
    for i in tqdm(range(8239, 8239 + 178)):
        path = Path("data/papers") / str(i) / "content.json"
        with open(path, "r") as bf:
            data: dict = json.load(bf)
            text = data["Full Text"]
        line_text = text.split("\n")

        current_block = []
        block_text = []
        for line in line_text:
            if line == "":
                block = " ".join(current_block)
                block_text.append(block)
                block_text.append("")
                current_block = []
            else:
                current_block.append(line)
        print("hey")
        # print(text)


if __name__ == "__main__":
    # abstract_and_text_to_json()
    clean_text()
