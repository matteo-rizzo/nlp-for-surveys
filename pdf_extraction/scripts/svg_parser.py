import os
import xml.etree.ElementTree as ET
from pathlib import Path


class SVGTree:
    def __init__(self, root: ET.Element):
        self.root: ET.Element = root
        self.current_index: int = 0
        self.nodes: dict[int, list[str]] = {}

    def add_node(self):
        index = self.current_index
        self.current_index += 1
        self.nodes[index] = []
        return index

    def add_to_node(self, index: int, text: str):
        self.nodes[index].append(text)


def get_tag(child):
    # Check if the child is an element
    if child.tag.startswith('{'):
        # Get the tag name without the namespace prefix
        return child.tag.split('}')[1]
    else:
        return child.tag


# I want to keep the tree information, though
def traverse(node: ET.Element, tree: SVGTree):
    # Check if the node has children
    children: list[ET.Element] = node.findall('*')
    if children:
        # Node has children, keep going
        # Recursively traverse the children
        tag: str = get_tag(node)
        if tag == "text":
            index = tree.add_node()
            for child in children:
                text_span = traverse(child, tree)
                tree.add_to_node(index, text_span)
        else:
            # Ignore output
            for child in children:
                traverse(child, tree)
    else:
        # Node is a leaf node, so print its text of tspan
        tag: str = get_tag(node)
        # The SVG <tspan> element defines a subtext within a <text> element or another <tspan> element.
        if tag == "tspan":
            # print(f"\t[{tag}] {node.text}")
            return node.text


def read_file(path):
    svg_tree = ET.parse(path)

    # Get the root element of the SVG tree
    svg_root: ET.Element = svg_tree.getroot()

    text_tree: SVGTree = SVGTree(svg_root)
    # Traverse the SVG tree
    traverse(text_tree.root, text_tree)


def main():
    # Load the SVG file
    svg_folder: Path = Path("data/papers/8239/svg")
    files = os.listdir(svg_folder)
    for file in files:
        svg_path: Path = svg_folder / str(file)
        read_file(svg_path)


if __name__ == "__main__":
    main()
