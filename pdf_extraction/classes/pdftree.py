import json
from pathlib import Path
from typing import Union

from pdf_extraction.classes.svgtree import SVGTree
from pdf_extraction.scripts.svg_parser import read_svg_files


class PDFTree:
    root_folder: Path = Path("data/papers")
    processed_folder: Path = Path("data/processed")

    def __init__(self, paper_id: Union[int, str]):
        # Paper id information
        if isinstance(paper_id, int):
            paper_id = str(paper_id)
        self.id: str = paper_id

        # Pathing
        svg_folder: Path = self.root_folder / paper_id / "svg"
        json_folder: Path = self.processed_folder / "success"

        # Add information gathered from svg file
        self.svg_tree: SVGTree = read_svg_files(svg_folder)
        self.svg_tree.merge_text_nodes()

        # Add information gathered from pdf miners
        json_path = json_folder / f"{paper_id}.json"
        if json_path.exists():
            with open(json_folder / f"{paper_id}.json", "r") as bf:
                self.json_info: dict = json.load(bf)
        else:
            self.json_info: dict = {"Error": "Unavailable pdf parsing"}


if __name__ == "__main__":
    tree = PDFTree(8239)
    print("")
