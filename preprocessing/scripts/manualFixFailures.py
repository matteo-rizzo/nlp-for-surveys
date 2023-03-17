import os
import shutil
from pathlib import Path

import rispy
from pypdfium2 import PdfDocument

from preprocessing.classes.paper import Paper
from preprocessing.scripts.pdfToJson import extract_from_name, compare_titles


def main():
    out_folder: Path = Path("data/processed")
    failures_folder: Path = out_folder / "failure"

    # Read bibliography file and extract info
    with open(out_folder / "updated_ris.ris", 'r', encoding="UTF-8") as bibliography_file:
        ris_bibliography: list[dict] = rispy.load(bibliography_file)

    paper_paths: list[Path] = [failures_folder / str(folders) for folders in os.listdir(failures_folder)]
    to_remove: list[Path] = []
    for p in paper_paths:
        # Generate exact path to pdf file
        pdf_file_name: str = str(os.listdir(p)[0])
        # Get paper name indicated in title
        pdf_title, author_year = extract_from_name(pdf_file_name)

        path_to_pdf: Path = p / pdf_file_name
        # Read pdf file with the library
        pdf_file: PdfDocument = PdfDocument(path_to_pdf)
        # Search for corresponding RIS file
        for idx, ris in enumerate(ris_bibliography):
            ris_title: str = ris["title"]
            title_similarity: float = compare_titles(pdf_title, ris_title)
            # Lower threshold for failures
            if title_similarity > 0.85:
                print("\n\n----------------------------------------------")
                print("  I've found the following similar titles (filename vs ris title):")
                print(f" - `{pdf_file_name}`\n - `{ris_title}`.")
                print("----------------------------------------------")
                print("But I coulnd't find the author. The RIS contains the following info:")
                # print(f"- Authors: {ris['authors']}\n- Year: {ris['year']}")
                if len(author_year) > 1:
                    tentative_year = int(author_year[1])
                    print("Extracted vs RIS comparison:")
                    print(f"- [{tentative_year}] VS [{ris['year']}]")
                    print(f"- [{author_year[0]}] VS {ris['authors']}")
                print("Here's a snippet of text:")
                print(f"{pdf_file[0].get_textpage().get_text_range()[:200]}")
                # ------------------------------------------------------------------------
                valid_input: bool = False
                stop_for: bool = False
                while not valid_input:
                    valid_input: bool = True
                    print("----------------------------------------------")
                    # response = "N"
                    response: str = input("Do you this is a match? (Y/N)")
                    if response == "Y" or response == "y":
                        print("Nice! Adding to successes.")
                        # Remove found ris
                        ris_bibliography.pop(idx)
                        # Save paper
                        paper: Paper = Paper(index=int(p.stem), pdf_file=pdf_file, ris=ris)
                        paper.to_json(containing_folder=out_folder / "success", original_name=pdf_file_name)
                        # Will be removed
                        to_remove.append(p)
                        stop_for: bool = True
                    elif response == "N" or response == "n":
                        print("Ok. Leaving as is.")
                    else:
                        print("Invalid input.")
                        valid_input: bool = False
                # ------------------------------------------------------------------------
                if stop_for:
                    break

        pdf_file.close()

    for path in to_remove:
        shutil.rmtree(path)


if __name__ == "__main__":
    main()
