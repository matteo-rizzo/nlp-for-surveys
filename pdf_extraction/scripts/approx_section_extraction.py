import json
from pathlib import Path

from pdfminer.high_level import extract_text
from tqdm import tqdm

from pdf_extraction.classes.pdftree import PDFTree


def abstract_and_text_to_json():
    for i in tqdm(range(8239, 9882)):
        # FIXME: CRASH AT "ORIGINAL_NAME"
        tree = PDFTree(i)
        path = Path("data/papers") / str(i)
        text = extract_text(path / tree.json_info['original_name'])
        data = {
            "Ris": tree.json_info["ris"],
            "Full Text": text
        }
        with open(path / f"content.json", 'w') as f:
            json.dump(data, f, indent=2)


# 1 - Find repeated blocks and remove them (done)
# 2 - Remove numbers (done)
# 3 - Remove \t and \r
# 4 - Remove strings which are just characters divided by spaces (unreadable)
# 5 - Remove short blocks (but what if they merge?)
# 6 - Remove strings with weird characters like copyright
# 7 - Remove references (from bottom)
# 8 - remove links and emails

# END - when join all together, clean again from garbage: DOIS, links, emails
def clean_text():
    # FIXME: CRASH AT "ORIGINAL_NAME"
    for i in tqdm(range(8239, 8239 + 178)):
        path = Path("data/papers") / str(i) / "content.json"
        with open(path, "r") as bf:
            data: dict = json.load(bf)
            text = data["Full Text"]
        line_text = text.split("\n")

        current_block = []
        block_text = []
        # --- merge blocks ----
        for line in line_text:
            if line == "":
                block = " ".join(current_block)
                block_text.append(block)
                block_text.append("[BLK]")
                current_block = []
            else:
                current_block.append(line)
        print("hey")
        # --- Find blocks which have duplicates (probably header / footers) ---
        seen = []
        to_remove: set[int] = set()
        for index, block in enumerate(block_text):
            if block in seen:
                to_remove.add(index)
                to_remove.add(seen.index(block))
            seen.append(block)
        # --- Only keep non duplicate blocks ---
        non_duplicate_text = []
        for index, block in enumerate(block_text):
            if index not in to_remove:
                # -- Also check it's not just a number (probably page number)
                if not block.strip().isdigit():
                    non_duplicate_text.append(block)
        print("hey")
        # print(text)


if __name__ == "__main__":
    # abstract_and_text_to_json()
    clean_text()
