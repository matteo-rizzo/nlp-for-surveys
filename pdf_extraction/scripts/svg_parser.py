import os
import xml.etree.ElementTree as ET
from pathlib import Path

from pdf_extraction.classes.svgtree import SVGTree


def get_tag(node: ET.Element) -> str:
    """
    Get tag for current XML tree node

    :param node: element of the XML tree
    :return: tag of the elemt
    """
    # Check if the node is an element
    if node.tag.startswith('{'):
        # Get the tag name without the namespace prefix
        return node.tag.split('}')[1]
    else:
        return node.tag


def traverse(node: ET.Element, tree: SVGTree):
    """
    Recursive function to traverse the tree

    :param node: current node being examined
    :param tree: the svg tree being build
    :return: text of the node, if found. Else nothing is returned
    """
    # Check if the node has children
    children: list[ET.Element] = node.findall('*')
    if children:
        # Node has children, keep going
        # Recursively traverse the children
        tag: str = get_tag(node)
        if tag == "text":
            # Add node text to tree
            index = tree.add_node()
            for child in children:
                text_span = traverse(child, tree)
                tree.add_to_node(index, text_span)
        else:
            # Ignore output
            for child in children:
                traverse(child, tree)
    else:
        # Node is a leaf node
        tag: str = get_tag(node)
        # The SVG <tspan> element defines a subtext within a <text> element or another <tspan> element.
        if tag == "tspan":
            return node.text


def read_svg_files(svg_folder) -> SVGTree:
    """
    Read all svg files in a folder (individual pages of pdf)
    :param svg_folder: path to a folder containing one or multiple svg files

    :return: SVGTree data structure containing extracted text
    """
    files = os.listdir(svg_folder)
    text_tree: SVGTree = SVGTree()
    for file in files:
        svg_path: Path = svg_folder / str(file)
        svg_tree = ET.parse(svg_path)

        # Get the root element of the SVG tree
        svg_root: ET.Element = svg_tree.getroot()

        # Traverse the SVG tree
        traverse(svg_root, text_tree)
    return text_tree

# def test():
#     # Load the SVG file
#     svg_folder: Path = Path("data/papers/8239/svg")
#     tree = read_svg_files(svg_folder)
#     print("")
#
#
# if __name__ == "__main__":
#     test()
