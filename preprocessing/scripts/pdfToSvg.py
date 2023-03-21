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
            output.add_page(input_pdf.pages[i])
            new_path = os.path.join(fpath.parent / 'svg', name + f"_p{i}.pdf")
            with open(new_path, "wb") as outputStream:
                output.write(outputStream)
            pdfs.append(new_path)

        return pdfs


def pdf_to_svg(file) -> None:
    args = ['C:/Program Files/Inkscape/bin/inkscape',
            '--without-gui',
            '--actions=export-type:svg;export-do',
            '--export-dpi=300',
            file]
    # Note: saving the return in a variable is necessary for its execution
    p = subprocess.Popen(args)
    p.communicate()
    p.terminate()


def main():
    papers_main_folder: Path = Path("data/papers")
    paper_paths: list[Path] = [papers_main_folder / str(folders) for folders in os.listdir(papers_main_folder)]

    # Go through pdfs
    for p in tqdm(paper_paths, desc="Reading pdf files"):
        pdf_file_name: str = str(os.listdir(p)[0])
        pdf_path: Path = p / pdf_file_name
        # pdf_path: Path = Path(p)
        (pdf_path.parent / "svg").mkdir(exist_ok=True)
        pdfs = split_pdf(pdf_path, pdf_path.stem)
        for pages in pdfs:
            pdf_to_svg(pages)
            os.remove(pages)

if __name__ == "__main__":
    main()
