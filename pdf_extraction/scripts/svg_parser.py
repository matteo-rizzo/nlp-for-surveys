import xml.etree.ElementTree as ET


def get_tag(child):
    # Check if the child is an element
    if child.tag.startswith('{'):
        # Get the tag name without the namespace prefix
        return child.tag.split('}')[1]
    else:
        return child.tag


def traverse(node):
    # Check if the node has children
    children = node.findall('*')
    if children:
        # Node has children, so print its tag name and attributes
        tag = get_tag(node)
        if tag == "text":
            print(tag, node.attrib)
        # Recursively traverse the children
        for child in children:
            traverse(child)
    else:
        # Node is a leaf node, so print its text of tspan
        tag = get_tag(node)
        if tag == "tspan":
            print(f"\t[{tag}] {node.text}")


def main():
    # Load the SVG file
    svg_file = 'data/papers/8239/svg/Hindawi - 2009 - 2020 A Publishing Odyssey_p0.svg'
    svg_tree = ET.parse(svg_file)

    # Get the root element of the SVG tree
    svg_root = svg_tree.getroot()

    # Traverse the SVG tree
    traverse(svg_root)


if __name__ == "__main__":
    main()
