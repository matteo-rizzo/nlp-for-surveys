class SVGTree:
    def __init__(self):
        self.current_index: int = 0
        self.nodes: dict = {}

    def add_node(self) -> int:
        """
        Add a "node" to the dictionary

        :return: index of node added
        """
        index = self.current_index
        self.current_index += 1
        self.nodes[index] = []
        return index

    def add_to_node(self, index: int, text: str) -> None:
        """
        Add text piece to a node of the dict

        :param index: index of node to which to add
        :param text: text piece to add
        """
        # TODO: raise exception when index not present
        self.nodes[index].append(text)

    def merge_text_nodes(self) -> None:
        """
        Joins the list of strings of nodes as a single block of text
        """
        for key, value in self.nodes.items():
            self.nodes[key] = " ".join(value)
