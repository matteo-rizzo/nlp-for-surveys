import os
import subprocess
from pathlib import Path

from PyPDF2 import PdfReader, PdfWriter


def split_pdf(fpath: Path, name: str) -> list[str]:
    with open(fpath, "rb") as buffer:
        input_pdf = PdfReader(buffer)

        pdfs = []
        for i in range(len(input_pdf.pages)):
            output = PdfWriter()
            output.add_page(input_pdf.pages[i])
            new_path = os.path.join(fpath.parent / 'tmp', name + f"_p{i}.pdf")
            with open(new_path, "wb") as outputStream:
                output.write(outputStream)
            pdfs.append(new_path)

        return pdfs


def pdf_to_svg(file) -> None:
    args = ['C:/Program Files/Inkscape/bin/inkscape', '--without-gui', '--actions=export-type:svg;export-do',
            '--export-dpi=300',
            file]
    subprocess.Popen(args)


def main():
    pdf_path: Path = Path("data/papers/8239/Hindawi - 2009 - 2020 A Publishing Odyssey.pdf")
    pdfs = split_pdf(pdf_path, pdf_path.stem)
    for p in pdfs:
        pdf_to_svg(p)
    # files = glob.glob('./data/papers/8239/' + '*.pdf', recursive=True)
    #
    # pdfs = split_pdf(pdf_path, svg_name, './data/papers/8239/tmp')
    #
    # if not os.path.exists(svg_path):
    #     os.makedirs(svg_path)
    #
    # for p in pdfs:
    #     pdf_to_svg(svg_path, p)
    #
    # for file in files:
    #     args = ['C:/Program Files/Inkscape/bin/inkscape', '--actions=export-type:svg;export-do', '--export-dpi=300',
    #             file]
    #     p = subprocess.Popen(args)


if __name__ == "__main__":
    main()
