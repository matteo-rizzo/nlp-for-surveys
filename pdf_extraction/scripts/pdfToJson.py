import glob
import os
import string
from pathlib import Path

import jellyfish
import rispy
from pypdfium2 import PdfDocument, PdfTextPage
from tqdm import tqdm

from pdf_extraction.classes.paper import Paper


# This is another nice library, but for now trying pdfium as main library
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


def find_author_in_page(page, author_name, ris):
    author_index: tuple[int, int] = page.search(author_name).get_next()
    # Author wasn't found, but there might be typos in the pdf file name. Check the RIS author
    if author_index is None:
        # Check ris author instead (first author is fine)
        ris_author: str = ris["authors"][0].split(",")[0]
        author_index: tuple[int, int] = page.search(ris_author).get_next()
        # Ris author found
        if author_index is not None:
            return True
        # Neither was found. Return false
        else:
            return False
    # Author from name was found
    else:
        return True


def main():
    papers_main_folder: Path = Path("data/papers")
    out_folder: Path = Path("data/processed")

    # Read bibliography file and extract info
    with open("data/TwinTransition_def_Ris.ris", 'r', encoding="UTF-8") as bibliography_file:
        ris_bibliography: list[dict] = rispy.load(bibliography_file)

    paper_paths: list[Path] = [papers_main_folder / str(folders) for folders in os.listdir(papers_main_folder)]

    # Go through pdfs
    for p in tqdm(paper_paths, desc="Reading pdf files"):
        # Generate exact path to pdf file
        pdf_file_name: str = Path(glob.glob(f"{p}/*.pdf")[0]).name
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
            title_similarity: float = compare_titles(pdf_title, ris_title)
            author_found: bool = False
            # Threshold seems decent
            if title_similarity > 0.95:
                # Likely to be a good match. But there are similar titles!!
                # Look for the author in the first pages
                author_name: str = author_year[0].split(" ")[0]  # Remove stuff like "et al"
                author_name: str = "".join(filter(lambda char: char in string.printable, author_name))  # Remove garbage
                # Search for author in first page (match is very likely)
                page_num: int = 0
                # Check first 3 pages [0, 1, 2] (if available)
                while page_num < min(3, len(pdf_file)) and not author_found:
                    page: PdfTextPage = pdf_file.get_page(page_num).get_textpage()
                    author_found: bool = find_author_in_page(page, author_name, ris)
                    page_num += 1
            # If author is confirmed
            if author_found:
                # Add ris
                paper_ris: dict = ris
                # Remove found ris
                ris_bibliography.pop(idx)
                # Stop searching
                break

        # ----------------------------------------------------------------------------------------------
        # Fail case
        if not paper_ris:
            # unresolved.append((pdf_file, pdf_title))
            # text = pdf_file[0].get_textpage().get_text_range()
            failure_folder: Path = out_folder / "failure" / p.stem
            failure_folder.mkdir(parents=True, exist_ok=True)
            pdf_file.save(failure_folder / pdf_file_name)
        else:
            paper: Paper = Paper(index=int(p.stem), pdf_file=pdf_file, ris=paper_ris)
            paper.to_json(containing_folder=out_folder / "success", original_name=pdf_file_name)
            # papers.append(paper)

    with open(out_folder / "updated_ris.ris", 'w', encoding="UTF-8") as bibliography_file:
        rispy.dump(ris_bibliography, bibliography_file)
    print("Done")


if __name__ == "__main__":
    main()

# Failures so far
# - Corrupted raw text
# - Weird letters in names 2
#
