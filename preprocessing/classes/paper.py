from typing import Dict

from pypdfium2 import PdfDocument


class Paper:
    def __init__(self, index: int, pdf_file: PdfDocument, ris: Dict):
        self.id: int = index
        self.reader: PdfDocument = pdf_file
        self.ris: Dict = ris

        self.metadata: dict = pdf_file.get_metadata_dict()
        self.raw_text: str = pdf_file[0].get_textpage().get_text_range()
        self.toc = []
        for item in pdf_file.get_toc():
            if item.n_kids == 0:
                state = "*"
            elif item.is_closed:
                state = "-"
            else:
                state = "+"
            self.toc.append((state, item.title))

    def print_toc(self):
        for item in self.reader.get_toc():
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
        title = self.reader.get_metadata_value('Title')
        return f"[{self.id}] {title if title else '(UNDEFINED)'}"
