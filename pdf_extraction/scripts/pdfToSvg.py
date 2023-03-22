import glob
import os
import subprocess
from pathlib import Path

from PyPDF2 import PdfReader, PdfWriter
from tqdm import tqdm


def split_pdf(fpath: Path, name: str) -> list[str]:
    with open(fpath, "rb") as buffer:
        input_pdf = PdfReader(buffer)

        pdfs = []
        for i in range(len(input_pdf.pages)):
            output = PdfWriter()
            try:
                output.add_page(input_pdf.pages[i])
                new_path = os.path.join(fpath.parent / 'svg', name + f"_p{i}.pdf")
                with open(new_path, "wb") as outputStream:
                    output.write(outputStream)
                pdfs.append(new_path)
            except AssertionError:
                print(f"Obj is none for [{fpath.parent.stem}] `{fpath.stem}` (page {i + 1})")

        return pdfs


def pdf_to_svg(file) -> None:
    # TODO: generalize
    args = ['C:/Program Files/Inkscape/bin/inkscape',
            '--without-gui',
            '--actions=export-type:svg;export-do',
            '--export-dpi=300',
            file]
    p = subprocess.Popen(args)
    # Wait for the process to terminate
    # TODO: could probably run multiple at the same time
    # p.communicate()
    # Terminate process if still running
    # p.terminate()


def all_to_svg():
    papers_main_folder: Path = Path("data/papers")
    paper_paths: list[Path] = [papers_main_folder / str(folders) for folders in os.listdir(papers_main_folder)]

    # Go through pdfs
    for p in tqdm(paper_paths, desc="Reading pdf files"):
        pdf_file_name: str = str(os.listdir(p)[0])
        pdf_path: Path = p / pdf_file_name
        (pdf_path.parent / "svg").mkdir(exist_ok=True)
        pdfs = split_pdf(pdf_path, pdf_path.stem)
        for pages in pdfs:
            pdf_to_svg(pages)


def delete_leftovers():
    papers_main_folder: Path = Path("data/papers")
    paper_paths: list[Path] = [papers_main_folder / str(folders) for folders in os.listdir(papers_main_folder)]
    for p in tqdm(paper_paths, desc="Looping through files"):
        inner_folder = p / "svg"
        pdfs = (glob.glob(f"{inner_folder}/*.pdf"))
        for to_del in pdfs:
            os.remove(to_del)


if __name__ == "__main__":
    all_to_svg()
    # NOTE: RUN THESE SEPARATELY!!
    # The calls to inkscape might finish later than the python script
    # delete_leftovers()
