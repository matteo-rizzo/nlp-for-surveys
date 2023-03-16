import os
import string
from pathlib import Path

import jellyfish
import rispy
from pypdfium2 import PdfDocument
from tqdm import tqdm

from preprocessing.classes.paper import Paper


# This is another nice library, but for now trying pdfium as main library
from PyPDF2 import PdfReader
# pypdf_test: PdfReader = PdfReader(path_to_pdf)
# text_pypdf = pypdf_test.pages[0].extract_text()

# Save json for each document

# Anche prevedere step supervisionato, save on the side the not sure
def compare_titles(title: str, ris_title: str) -> float:
    # Ris titles are usually complete, titles on pdfs are not
    cut_ris_title = ris_title[:len(title)]
    # Evaluate similarity with extracted title
    # Jaro-Winkler is a modification/improvement to Jaro distance (string-edit distance),
    # like Jaro it gives a floating point response in [0,1]
    # where 0 represents two completely dissimilar strings and 1 represents identical strings.
    title_similarity = jellyfish.jaro_winkler_similarity(title, cut_ris_title)
    return title_similarity


def extract_from_name(pdf_file_name: str) -> [str, list]:
    *author_year, name = [x.strip() for x in pdf_file_name.split(" - ")]

    # Extract an approximate name from the file name (Usually in format author - year - title.pdf)
    # Sometimes year is absent. Regardless, title is always rightmost and has a .pdf suffix
    *name, file_format = name.rsplit(".")
    title = "".join(name)
    # Clean garbage
    title = "".join(filter(lambda char: char in string.printable, title))
    assert file_format == "pdf", "Suffix should always be pdf"
    return title, author_year


def main():
    papers_main_folder: Path = Path("data/papers")

    # Read bibliography file and extract info
    with open("data/TwinTransition_def_Ris.ris", 'r', encoding="UTF-8") as bibliography_file:
        ris_bibliography: list[dict] = rispy.load(bibliography_file)

    paper_paths: list[Path] = [papers_main_folder / str(folders) for folders in os.listdir(papers_main_folder)]
    papers: list[Paper] = []

    # Go through pdfs
    unresolved: list[tuple[PdfDocument, str]] = []
    failed_counter: int = 0
    for p in tqdm(paper_paths, desc="Reading pdf files"):
        # Generate exact path to pdf file
        pdf_file_name: str = str(os.listdir(p)[0])
        # Get paper name indicated in title
        pdf_title, author_year = extract_from_name(pdf_file_name)

        path_to_pdf: Path = p / pdf_file_name
        # Read pdf file with the library
        pdf_file: PdfDocument = PdfDocument(path_to_pdf)
        # ----------------------------------------------------------------------------------------------
        # Search for corresponding RIS file
        paper_ris = {}
        for idx, ris in enumerate(ris_bibliography):
            ris_title: str = ris["title"]
            title_similarity = compare_titles(pdf_title, ris_title)
            sad_flag: bool = False
            # Threshold seems decent
            if title_similarity > 0.95:
                # Likely to be a good match. But there are similar titles!!
                # Look for the author in the first page
                author_name = author_year[0].split(" ")[0]  # Remove stuff like "et al"
                author_name = "".join(filter(lambda char: char in string.printable, author_name))
                page = pdf_file.get_page(0).get_textpage()
                author_index = page.search(author_name).get_next()
                if author_index is None:
                    # Check ris author instead
                    ris_author = ris["authors"][0].split(",")[0]
                    author_index = page.search(ris_author).get_next()
                    if author_index is not None:
                        sad_flag = True
                    else:
                        failed_counter += 1
                        # PdfDocument(path_to_pdf).get_page(0).get_textpage().get_text_range(0)
                        # pypdf_test: PdfReader = PdfReader(path_to_pdf)
                else:
                    sad_flag = True
            if sad_flag:
                # Add ris
                paper_ris = ris
                # Remove found ris
                ris_bibliography.pop(idx)
                # Stop searching
                break

        # ----------------------------------------------------------------------------------------------
        # Search for doi within document, going through each page until found
        # page_index: int = 0
        # doi_found: bool = False
        # while page_index < len(pdf_file) and not doi_found:
        #     # Search word occurrence - result will be the index of text within document
        #     doi_index = pdf_file.get_page(page_index).get_textpage().search("doi", match_whole_word=True).get_next()
        #     # Search within the page, starting from the index found, for the doi
        #     if doi_index is not None:
        #         page = pdf_file.get_page(page_index)
        #         doi = page.get_textpage().get_text_range(doi_index[0])
        #         regex = r"\b(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?![\"&\'])\S)+)\b"
        #         matches = re.search(regex, doi, re.IGNORECASE)
        #         # This is still tentative, could be a reference
        #         if matches is not None:
        #             doi_found = True
        #
        #             discovered_doi = matches.group(0)
        #             # print(f" - {discovered_doi}")
        #             metrics["success"] += 1
        #         # doi regex was not succesful
        #         else:
        #             # print(f" ! regex failed for [{pdf_file_name}]")
        #             metrics["regex_failed"] += 1
        #             # failures["regex_failed"].append(pdf_file)
        #     # Increase
        #     page_index += 1
        #     if page_index == len(pdf_file) and not doi_found:
        #         # print(f" ! doi not found for document [{pdf_file_name}]")
        #         metrics["doi_not_found"] += 1
        #         failures.append(Paper(index=int(p.stem), pdf_file=pdf_file, ris={}))

        if not paper_ris:
            unresolved.append((pdf_file, pdf_title))
        else:
            paper: Paper = Paper(index=int(p.stem), pdf_file=pdf_file, ris=paper_ris)

            papers.append(paper)

    print("Done")
    print(f"Failed: {failed_counter}")


if __name__ == "__main__":
    main()
