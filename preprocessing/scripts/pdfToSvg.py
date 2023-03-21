"""
Author: Vitchyr Pong
Convert a PDF to svg files. Note that this is pretty slow since it makes
subprocess calls to inkscape's PDF-to-SVG command line convert.
Requirements:
    - Inkscape (https://inkscape.org/)
    - pyPdf (http://pybrary.net/pyPdf/)
    - A '/tmp' directory. If not, you must pass in another directory.
        Use 'python pdf2svg.py -h' for more information
Example Usage:
    $ python pdf2svgs.py path/to/foo.pdf path/to/svgs new_svg_name
will result in the new files:
    path/to/svgs/new_svg_name_all_p0.svg
    path/to/svgs/new_svg_name_all_p1.svg
    etc.
"""
import argparse
import os
import subprocess

from PyPDF2 import PdfReader, PdfWriter


def split_pdf(fpath, name, tmp_dir):
    """
    Split a pdf into multiple PDFs, one per page.
    Parameters
    ----------
    fpath : string
        Path to a PDF.
    name : string
        Base name for the SVGs.
    tmp_dir: string
        A directory where temporary PDFs are saved
    Return
    ------
    pdfs: list of strings
        A list of path directories to the temporary PDFs, in order.
    """
    input_pdf = PdfReader(open(fpath, "rb"))

    directory = os.path.dirname(fpath)

    pdfs = []
    for i in range(len(input_pdf.pages)):
        output = PdfWriter()
        output.add_page(input_pdf.pages[i])
        new_path = os.path.join(tmp_dir, name + "_p{0}.pdf".format(i))
        with open(new_path, "wb") as outputStream:
            output.write(outputStream)
        pdfs.append(new_path)

    return pdfs


def file_name(path):
    """ Get the name of a file without the extension. """
    return os.path.split(path)[-1].split(".")[0]


def pdf_to_svg(out_dir, fpath):
    """
    Convert a pdf to an svg.
    Parameters
    ----------
    out_dir : string
        Directory where to save the SVG.
    fpath : string
        Path to a PDF. The SVG will have the same name as this but with a .svg extension.
    """
    out_name = file_name(fpath) + ".svg"
    out_fpath = os.path.join(out_dir, out_name)
    subprocess.call(["inkscape", "-l", out_fpath, fpath])


def main():
    parser = argparse.ArgumentParser(description="Convert a PDF to svg files.")

    # parser.add_argument("pdf_path", help="Path to the pdf.")
    # parser.add_argument("svg_dir", help="Directory to save the svg.")
    # parser.add_argument("svg_name", help="Base name of the svgs.")
    parser.add_argument("-t",
                        "--tmp_dir",
                        help="Where to save temporary PDFs.",
                        default="data/tmp")

    args = parser.parse_args()
    pdf_path = "data/papers/8239/Hindawi - 2009 - 2020 A Publishing Odyssey.pdf"
    svg_path = "data/papers/8239"
    svg_name = "test"
    pdfs = split_pdf(pdf_path, svg_name, args.tmp_dir)

    if not os.path.exists(svg_path):
        os.makedirs(svg_path)

    for p in pdfs:
        pdf_to_svg(svg_path, p)


if __name__ == '__main__':
    main()
