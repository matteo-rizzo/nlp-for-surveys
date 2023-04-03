import json
import re
from pathlib import Path
from typing import Optional

from pdfminer.high_level import extract_text
from tqdm import tqdm

from pdf_extraction.classes.pdftree import PDFTree


def abstract_and_text_to_json() -> None:
    """
    Turns the previously associated RIS + paper combinations into a simplified RIS + full text json
    """
    # VERY SLOW - THERE ARE SOME FAILURE CASES
    # TODO: fix manual range

    # --- Go through all documents ---
    for i in tqdm(range(8239, 9882)):
        # Associates documents and RIS files
        tree: PDFTree = PDFTree(i)
        path: Path = Path("data/papers") / str(i)
        # If the name exists (it only doesn't if the original RIS + paper failed
        if name := tree.json_info.get('original_name'):
            try:
                # Extract text with PDFMiner
                text = extract_text(path / name)
                data = {
                    "Ris": tree.json_info["ris"],
                    "Full Text": text
                }
                # Save to file
                with open(path / f"content.json", 'w') as f:
                    json.dump(data, f, indent=2)
            except AssertionError:
                # Sometimes text extraction can fail (rare)
                continue
        # Skip failure cases
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

def has_copyright_chars(block: str) -> bool:
    # Check if it has weird characters (like copyright, usually only contained in legal headers/footers)
    return bool(re.search(r'©', block))


def is_just_number(block: str) -> bool:
    # Check it's not just a number (probably page number)
    return block.strip().isdigit()


def has_doi(block: str) -> bool:
    # Check if a doi is present (dois are basically never in the body text)
    return bool(re.search(r'\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', block))


def has_email(block: str) -> bool:
    # Check if the block has an email address
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    return bool(email_pattern.search(block))


def has_url(block: str) -> bool:
    # Check if there's an url in the block
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return bool(url_pattern.search(block))


def is_alternation(block: str) -> bool:
    # For when blocks are messy evenly spaced characters
    # e.g. i n t r o d u c t i o n
    return bool(re.match(r'^([\w,.!?*)(\-:<>+&/–;] )*[\w,.!?*)(\-:<>+&/–;]$', block.strip()))


def is_number_alternation(block: str) -> bool:
    # Special case for things like 1 3 2 3 1 3.2 12.3
    # Usually poorly read tables
    return bool(re.match(r'^(\d+(\.\d+)? )*\d+(\.\d+)?$', block.strip()))


def is_figure(block: str) -> bool:
    # Check if block is just a Figure caption
    return bool(re.match(r'(figure \d+)', block.strip().lower()))


def is_page(block: str) -> bool:
    # Check special case of "Page x of x"
    return bool(re.match(r'(page \d+ of \d+)', block.strip().lower()))


def is_of(block: str) -> bool:
    # Check special case "x of x"
    return bool(re.match(r'(\d+ of \d+)', block.strip().lower()))


def has_no_chars(block: str) -> bool:
    pattern = r'^[\d\W]+$'
    match = re.match(pattern, block)
    return bool(match)


def check_condictions(block: str) -> bool:
    # A very unoptimized conditional case, but it's fast so it doesn't matter
    return has_copyright_chars(block) or is_just_number(block) \
        or has_doi(block) or has_url(block) or block.isupper() or \
        has_email(block) or is_alternation(block) or is_figure(block) \
        or is_page(block) or is_number_alternation(block) or has_no_chars(block) or is_of(block) or block == "" or len(block) <= 2


def find_references(strings_list: list[str]) -> Optional[int]:
    """
    Find the first (from the bottom) occurrence of the word "reference" in a given block of strings

    Args:
        strings_list: list of blocks (strings)

    Returns: index of first block from the bottom containing the word "references"
    """
    # Find the first
    for i in range(len(strings_list) - 1, -1, -1):
        if "references" in strings_list[i].lower():
            return i
    return None


def clean_block(block: str) -> str:
    """
    Clean a block from spurious information (weird characters, spaces and such)

    Args:
        block: block (string) to clean

    Returns: clean str
    """
    # spurious (cid:d+) encodings (not read properly)
    clean: str = re.sub(r'\(cid:\d+\)', '', block)
    # Uninformative common words
    words = ["Fig", "Fig.", "Figure", "Table", "Tab", "Tab."]
    clean = re.sub(r'\b(?:{})\b'.format('|'.join(map(re.escape, words))), '', clean, flags=re.IGNORECASE)
    # Weird characters, tabs, newpages
    clean: str = re.sub(r'[\t\r\f●…✓•→]', '', clean)
    # 3 or more "_"
    clean: str = re.sub(r"_{3,}", "", clean)
    # Dates
    clean: str = re.sub(r"\b\d{2}/\d{2}/\d{2}\b", "", clean)
    # References / bracketed numbers
    clean: str = re.sub(r"\[\s*(\d+(?:[ -–]\d+)?)\s*(?:,\s*(\d+(?:[ -–]\d+)?\s*(?:,\s*\d+(?:[ -–]\d+)?)*))?\s*]", "",
                        clean)
    # Sample number
    clean: str = re.sub(r"(N\s*=\s*\d+)", "", clean)
    # Empty brackets
    clean: str = re.sub(r'[\[\]()]', "", clean)
    # Special spaces
    clean: str = re.sub(r"[\u00A0\u2002]", " ", clean)
    # Excessive spaces
    clean: str = re.sub(r'\s+', ' ', clean).strip()
    return clean.strip()


def clean_text() -> None:
    """
    Main starting point for the text cleaning script (requires json simplified papers)
    """
    # TODO: fix manual ranges
    for i in tqdm(range(8239, 9882)):
        path: Path = Path("data/papers") / str(i) / "content.json"
        # Failure cases (~10)
        if not path.exists():
            continue
        with open(path, "r") as bf:
            data: dict = json.load(bf)
            text = data["Full Text"]
        line_text = text.split("\n")

        current_block: list[str] = []
        block_text: list[str] = []
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
            # Keep references
            if block in seen and block.lower() != "references":
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
        references_index: int = find_references(non_duplicate_text)
        data: dict = {
            "Ris": data["Ris"],
            "Full Text": non_duplicate_text if not references_index else non_duplicate_text[:references_index]
        }
        with open(path.parent / f"clean_content.json", 'w', encoding='UTF-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # abstract_and_text_to_json()
    clean_text()
