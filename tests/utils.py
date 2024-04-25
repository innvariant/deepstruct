import networkx as nx

from matplotlib import pyplot as plt


def plot_graph(graph, title):
    labels = nx.get_node_attributes(graph, "name")
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(
        graph,
        labels=labels,
        with_labels=True,
        node_size=700,
        node_color="lightblue",
        font_size=8,
        ax=ax,
    )
    plt.title(title)
    plt.show()


def calculate_network_metrics(graph):
    metrics = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "avg degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes(),
        "density": nx.density(graph),
        "Average cluster coefficient": nx.average_clustering(graph),
    }
    return metrics
