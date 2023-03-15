import os
from pathlib import Path
from typing import List

import rispy
from pypdfium2 import PdfDocument
from tqdm import tqdm

from preprocessing.classes.paper import Paper
import regex as re

# This is another nice library, but for now trying pdfium as main library
# from PyPDF2 import PdfReader
# pypdf_test: PdfReader = PdfReader(path_to_pdf)
# text_pypdf = pypdf_test.pages[0].extract_text()

def main():
    papers_main_folder: Path = Path("data/papers")
    with open("data/TwinTransition_def_Ris.ris", 'r', encoding="UTF-8") as bibliography_file:
        ris_bibliography = rispy.load(bibliography_file)

    paper_paths = [papers_main_folder / str(folders) for folders in os.listdir(papers_main_folder)]

    papers: List = []
    # for p in tqdm(paper_paths, desc="Reading pdf files"):
    for p in paper_paths:
        pdf_file_name: str = str(os.listdir(p)[0])
        path_to_pdf: Path = p / pdf_file_name

        pdf_file: PdfDocument = PdfDocument(path_to_pdf)

        for page_index in range(len(pdf_file)):
            # list of bounding boxes of the form (left, bottom, right, top)
            doi_bb = pdf_file.get_page(page_index).get_textpage().search("doi", match_whole_word=True).get_next()
            if doi_bb is not None:
                page = pdf_file.get_page(page_index)
                doi = page.get_textpage().get_text_range(doi_bb[0])
                regex = r"\b(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?![\"&\'])\S)+)\b"
                matches = re.search(regex, doi, re.IGNORECASE)
                if matches is not None:
                    doi_found = matches.group(0)
                    print(f" - {doi_found}")
                else:
                    print(f" ! doi not found for document [{pdf_file_name}]")
                    print("!")

        paper: Paper = Paper(index=int(p.stem), pdf_file=pdf_file, ris={})

        papers.append(paper)

    print("yeet")


if __name__ == "__main__":
    main()
