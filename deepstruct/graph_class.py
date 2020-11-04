class GraphClass:
    __slots__ = ['node_id', 'from_node', 'to_node', 'layer_id', 'weight']

    def __init__(self, node_id, layer_id, from_node, to_node,weight):
        self.node_id = node_id
        self.from_node = from_node
        self.to_node = to_node
        self.layer_id = layer_id
        self.weight = weight
