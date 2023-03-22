import json
from pathlib import Path
from typing import Dict

from pypdfium2 import PdfDocument


class Paper:
    def __init__(self, index: int, pdf_file: PdfDocument, ris: Dict):
        self.id: int = index
        self.pdf_file: PdfDocument = pdf_file
        self.ris: Dict = ris

        self.metadata: dict = pdf_file.get_metadata_dict()
        # self.raw_text: str = pdf_file[0].get_textpage().get_text_range()
        self.toc = []
        for item in pdf_file.get_toc():
            if item.n_kids == 0:
                state = "*"
            elif item.is_closed:
                state = "-"
            else:
                state = "+"
            self.toc.append((state, item.title))

    def get_raw_text(self):
        raw_text: str = ""
        for page in self.pdf_file:
            raw_text += page.get_textpage().get_text_range()
        return raw_text

    def get_toc(self):
        toc = []
        for item in self.pdf_file.get_toc():
            if item.n_kids == 0:
                state = "*"
            elif item.is_closed:
                state = "-"
            else:
                state = "+"
            toc.append((state, item.title))
        return toc

    def to_json(self, containing_folder: Path = Path("data/processed/success"), original_name: str = ""):
        containing_folder.mkdir(parents=True, exist_ok=True)

        data: dict = {
            "original_name": original_name,
            "ris": self.ris,
            "toc": self.toc,
            "text": self.get_raw_text()

        }
        with open(containing_folder / f"{self.id}.json", 'w') as f:
            json.dump(data, f, indent=2)

    def print_toc(self):
        for item in self.pdf_file.get_toc():
            if item.n_kids == 0:
                state = "*"
            elif item.is_closed:
                state = "-"
            else:
                state = "+"

            if item.page_index is None:
                target = "?"
            else:
                target = item.page_index + 1

            print(
                "    " * item.level +
                f"[{state}] {item.title} -> {target}  # {item.view_mode} {[round(c) for c in item.view_pos]}"
            )

    def __repr__(self):
        return f"[{self.id}] `{self.ris['title']}`"
