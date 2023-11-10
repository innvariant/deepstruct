import networkx


class LayeredGraph:

    def __init__(self):
        self.graph = networkx.DiGraph()
        self.vertex_id = 0
        self.layer_before = []

    def add_vertex(self, name):
        self.graph.add_node(self.vertex_id, name=name)
        if len(self.layer_before) > 0:
            for vertex in self.layer_before:
                self.graph.add_edge(vertex, self.vertex_id)
        self.layer_before = [self.vertex_id]
        self.vertex_id += 1

    def add_vertices(self, name, count):
        current_vertices = []
        for i in range(count):
            self.graph.add_node(self.vertex_id, name=name)
            current_vertices.append(self.vertex_id)
            self.vertex_id += 1
        if len(self.layer_before) > 0:
            for vertex in self.layer_before:
                for new_vertex in current_vertices:
                    self.graph.add_edge(vertex, new_vertex)
        self.layer_before = current_vertices

    def add_edge(self, src, dest):
        self.graph.add_edge(src, dest)






