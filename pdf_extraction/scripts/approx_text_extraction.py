import json
import re
from pathlib import Path

from pdfminer.high_level import extract_text
from tqdm import tqdm

from pdf_extraction.classes.pdftree import PDFTree


def abstract_and_text_to_json():
    # VERY SLOW - THERE ARE SOME FAILURE CASES
    for i in tqdm(range(8239, 9882)):
        tree = PDFTree(i)
        path = Path("data/papers") / str(i)
        if name := tree.json_info.get('original_name'):
            try:
                text = extract_text(path / name)
                data = {
                    "Ris": tree.json_info["ris"],
                    "Full Text": text
                }
                with open(path / f"content.json", 'w') as f:
                    json.dump(data, f, indent=2)
            except AssertionError:
                continue

        else:
            continue


# 1 - Find repeated blocks and remove them (done)
# 2 - Remove numbers (done)
# 3 - Remove \t and \r (done)
# 4 - Remove strings which are just characters divided by spaces (done)
# 5 - Remove short blocks (but what if they merge?)
# 6 - Remove strings with weird characters like copyright (done)
# 7 - Remove references (from bottom) (done)
# 8 - remove blocks with links and emails (done)

def has_copyright_chars(block):
    # Check if it has weird characters (like copyright, is usually header/footer
    return bool(re.search(r'©', block))


def is_just_number(block):
    # Check it's not just a number (probably page number)
    return block.strip().isdigit()


def has_doi(block):
    return bool(re.search(r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', block))


def has_email(block):
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    return bool(email_pattern.search(block))


def has_url(block):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return bool(url_pattern.search(block))


def is_alternation(block):
    # For when blocks are messy evenly spaced characters
    return bool(re.match(r'^([\w,.!?*)(\-:<>+&/] )*[\w,.!?*)(\-:<>+&/]$', block.strip()))


def is_number_alternation(block):
    return bool(re.match(r'^(\d+(\.\d+)? )*\d+(\.\d+)?$', block.strip()))


def is_figure(block):
    return bool(re.match(r'(figure \d+)', block.strip().lower()))


def is_page(block):
    return bool(re.match(r'(page \d+ of \d+)', block.strip().lower()))


def check_condictions(block):
    return has_copyright_chars(block) or is_just_number(block) \
        or has_doi(block) or has_url(block) or has_email(block) or is_alternation(block) or is_figure(block) \
        or is_page(block) or is_number_alternation(block)


def find_references(strings_list):
    for i in range(len(strings_list) - 1, -1, -1):
        if "references" in strings_list[i].lower():
            return i
    return None


def clean_block(block):
    # clean spurious (cid:d+) encodings (not read properly)
    clean = re.sub(r'\(cid:\d+\)', '', block)
    clean = re.sub(r'[\t\r\f●…•]', '', clean)
    clean = re.sub(r"_{3,}", "", clean)
    # clean = clean.replace('\u00A0', ' ')
    clean = re.sub(r"[\u00A0\u2002]", " ", clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean.strip()


def clean_text():
    for i in tqdm(range(8239, 9882)):
        path = Path("data/papers") / str(i) / "content.json"
        # Failure cases (~10)
        if not path.exists():
            continue
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
            if index not in to_remove and not check_condictions(block):
                block = clean_block(block)
                # CHECK AGAIN!
                if not check_condictions(block):
                    non_duplicate_text.append(block)
        # print("hey")
        references_index = find_references(non_duplicate_text)
        data = {
            "Ris": data["Ris"],
            "Full Text": non_duplicate_text if not references_index else non_duplicate_text[:references_index]
        }
        with open(path.parent / f"clean_content.json", 'w', encoding='UTF-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        # print(text)


if __name__ == "__main__":
    # abstract_and_text_to_json()
    clean_text()
