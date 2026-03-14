"""
Graphs
"""
import sys
import inspect
import math
import networkx as nx
import dgl
import numpy as np
import os
import torch
from collections import defaultdict

from .hyperparameters import HyperParameters
from . import datasets


def _isconnected(graph):
    return nx.is_strongly_connected(dgl.to_networkx(graph))


def _buckleup(graph, exist_ok=False):
    """
    Adds self loops to given graph.
    """
    if not exist_ok:
        # The number of edges in the given graph
        count = len(graph.edges())
        # Remove all self-loops in the graph, if present
        graph = dgl.remove_self_loop(graph)
        # Assert |E'| = |E|
        assert len(graph.edges()) == count
    # Add self-loops for each node in the graph and return a new graph
    return dgl.transforms.add_self_loop(graph)


def sample_(selfloop=True):
    """
    Returns a sample 6-node graph.
    """
    src = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5])
    dst = np.array([2, 5, 2, 3, 4, 5, 0, 1, 3, 5, 1, 2, 4, 5, 1, 3, 0, 1, 2, 3])

    graph = dgl.graph((src, dst))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def sample(params):
    """
    Returns a sample 6-node graph from hyper-parameters.
    """
    # Update network size
    params.size = 6
    return sample_(selfloop=params.selfloop)


def wheel_(size, selfloop=True):
    """
    Returns an undirected wheel graph.
    """
    # Check network size
    assert size > 1
    # Get graph from networkx
    graph = dgl.from_networkx(nx.wheel_graph(size))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def wheel(params):
    """
    Returns an undirected wheel graph from hyper-parameters.
    """
    return wheel_(params.size, selfloop=params.selfloop)


def cycle_(size, directed=False, selfloop=True):
    """
    Returns a cycle graph.
    """
    # Check network size
    assert size > 1
    # Get networkx constructor
    constructor = nx.DiGraph if directed else nx.Graph
    # Get graph from networkx
    graph = dgl.from_networkx(nx.cycle_graph(size, create_using=constructor))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def cycle(params):
    """
    Returns a cycle graph from hyper-parameters.
    """
    return cycle_(params.size, directed=params.directed, selfloop=params.selfloop)


def star_(size, selfloop=True):
    """
    Returns an undirected star graph.
    """
    # Check network size
    assert size > 1
    # Get graph from networkx. The graph has n + 1 nodes for integer n, so substract 1
    graph = dgl.from_networkx(nx.star_graph(size - 1))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def star(params):
    """
    Returns an undirected star graph.
    """
    return star_(params.size, selfloop=params.selfloop)


def line_(size, directed=False, selfloop=True):
    """
    Return a line graph.
    """
    # Check network size
    assert size > 1
    # Get networkx constructor
    constructor = nx.DiGraph if directed else nx.Graph
    # Get graph from networkx
    graph = dgl.from_networkx(nx.path_graph(size, create_using=constructor))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def line(params):
    """
    Return a line graph from hyper-parameters.
    """
    return line_(params.size, directed=params.directed, selfloop=params.selfloop)


def grid_(size, selfloop=True):
    """
    Returns a 2-D square grid graph.
    """
    # Check network size
    assert size > 1
    # Check network size is a perfect square (approximate solution)
    assert size == math.pow(int(math.sqrt(size) + 0.5), 2)
    rows = columns = int(math.sqrt(size))
    # Get graph from networkx
    graph = dgl.from_networkx(nx.grid_2d_graph(rows, columns))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def grid(params):
    """
    Returns a 2-D square grid graph from hyper-parameters.
    """
    return grid_(params.size, selfloop=params.selfloop)


def random_(size, probability, tries=100, seed=None, directed=False, selfloop=True):
    """
    Returns an Erdos-Renyi graph.
    """
    # Check network size
    assert size > 1
    # If seed is not set, use NumPy's global RNG
    if not seed:
        seed = np.random
    attempt = 0
    success = False
    while True:
        attempt += 1
        # Get graph from networkx
        graph = dgl.from_networkx(
            nx.erdos_renyi_graph(size, probability, seed=seed, directed=directed)
        )
        if _isconnected(graph):
            # Connected graph found; exit loop
            success = True
            break
        else:
            # Probability p should be greater than ((1 + e)ln(size))/size
            _p = math.log(size) / size
            if probability > _p and attempt < tries:
                # Probability is set correctly; permit failed attempts
                continue
            else:
                # Most likely all attempts will fail; exit loop
                break
    if not success:
        msg = f"Graph G({size, probability}) is disconnected. Try p > {_p}"
        raise Exception(msg)
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def random(params):
    """
    Returns an Erdos-Renyi graph from hyper-parameters
    """
    return random_(
        params.size,
        params.random.probability,
        tries=params.random.tries,
        seed=params.random.seed,
        directed=params.directed,
        selfloop=params.selfloop,
    )


def complete_(size, selfloop=True):
    """
    Returns an undirected fully-connected graph.
    """
    # Check network size
    assert size > 1
    # Get graph from networkx
    graph = dgl.from_networkx(nx.complete_graph(size))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def complete(params):
    """
    Returns an undirected fully-connected graph from hyper-parameters.
    """
    return complete_(params.size, selfloop=params.selfloop)


def karate_(selfloop=True):
    """
    Returns Zachary's Karate club social network.
    """
    # Get graph from networkx
    graph = dgl.from_networkx(nx.karate_club_graph())
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def karate(params):
    """
    Returns Zachary's Karate club social network from hyper-parameters.
    """
    # Update network size
    params.size = 34
    return karate_(selfloop=params.selfloop)


def wattsstrogatz_(
    size, knn, probability, tries=100, seed=None, selfloop=True
):  # pylint: disable=too-many-arguments
    """
    Returns a connected Watts–Strogatz small-world graph.
    """
    # Check network size
    assert size > 1
    # Check neighbourhood size
    assert knn > 1
    # If seed is not set, use NumPy's global RNG
    if not seed:
        seed = np.random
    # Get graph from networkx
    graph = dgl.from_networkx(
        nx.connected_watts_strogatz_graph(
            size, knn, probability, tries=tries, seed=seed
        )
    )
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def wattsstrogatz(params):
    """
    Returns a connected Watts–Strogatz small-world graph from hyper-parameters.
    """
    return wattsstrogatz_(
        params.size,
        params.wattsstrogatz.knn,
        params.wattsstrogatz.probability,
        tries=params.wattsstrogatz.tries,
        seed=params.wattsstrogatz.seed,
        selfloop=params.selfloop,
    )


def barabasialbert_(size, attachments, seed=None, selfloop=True):
    """
    Returns a random graph according to the Barabasi–Albert preferential attachment model.
    """
    # Check network size
    assert size > 1
    # Check neighbourhood size
    assert attachments > 0
    # If seed is not set, use NumPy's global RNG
    if not seed:
        seed = np.random
    # Get graph from networkx
    graph = dgl.from_networkx(nx.barabasi_albert_graph(size, attachments, seed=seed))
    # Try adding self-loops
    if selfloop:
        graph = _buckleup(graph)
    return graph


def barabasialbert(params):
    """
    Returns a random graph according to the Barabasi–Albert
    preferential attachment model from hyper-parameters.
    """
    return barabasialbert_(
        params.size,
        params.barabasialbert.attachments,
        seed=params.barabasialbert.seed,
        selfloop=params.selfloop,
    )


def snap(params):
    """
    Returns a SNAP dataset, identified by `params.snap.name`.
    """
    from .datasets import snap as snp
    graph = snp.getbyname(params.snap.name).read()
    # Update network size
    params.size = graph.num_nodes()
    return graph


def ogb(params):
    """
    Returns an OGB dataset, identified by `params.ogb.name`.
    Current only one dataset is supported, ogbl-collab.
    """
    assert params.ogb.name and isinstance(params.ogb.name, str)
    assert params.ogb.name.lower == "collab"
    dataset = datasets.ogb.Collab()
    graph = dataset.read()
    # Update network size
    params.size = graph.num_nodes()
    return graph


def francisbacon(params):
    """
    Returns a Six Degrees of Francis Bacon Network.
    The GML file of the graph must be generated manually because the
    dataset is dynamic. See the sixdegreesoffrancisbacon notebook in
    the scripts folder.
    """
    from .datasets import francisbacon
    dataset = francisbacon.FrancisBacon()
    graph = dataset.read()
    # Try adding self-loops
    if params.selfloop:
        graph = _buckleup(graph)
    params.size = graph.num_nodes()
    return graph


def gml(params):
    """
    Loads a custom GML graph from a path identified by `params.gml.path`.
    A custom name can be given for the graph using `params.gml.name`
    Directed graphs should be specified by setting `params.gml.directed` to True
    """
    assert params.gml.name, "gml.name GML network name not specified"
    assert params.gml.path, "gml.path GML file not specified"

    # Resolve GML file
    gml_file = os.path.abspath(os.path.expanduser(params.gml.path))
    assert os.path.isfile(gml_file), "GML file not found"

    # Load graph from GML file
    G = nx.read_gml(gml_file, destringizer=int)

    # Load graph using edge list so that we preserve node ids
    edges = [(u, v) for u, v, _ in nx.to_edgelist(G)]
    if not edges:
        raise RuntimeError("GML file loaded but contains no edges.")

    # Remap node IDs to integers
    nodes = {}
    src, dst = [], []
    for u, v in edges:
        if u not in nodes:
            nodes[u] = len(nodes)
        if v not in nodes:
            nodes[v] = len(nodes)
        src.append(nodes[u])
        dst.append(nodes[v])

    # Build DGL graph
    graph = dgl.graph((src, dst), num_nodes=len(nodes))
    if not params.gml.directed:
        graph = dgl.to_bidirected(graph)

    # Store original IDs (index i -> original node id)
    graph.gml_id = list(nodes.keys())

    # -------------------------
    # COPY NODE ATTRIBUTES (general)
    # Assumption: all node-attribute values you care about are already numeric
    # -------------------------
    inv = {new_id: orig_id for orig_id, new_id in nodes.items()}  # new_id -> original_id

    # collect all attribute names appearing on any node
    attr_names = set()
    for orig_id in inv.values():
        attr_names.update(G.nodes[orig_id].keys())

    for attr in sorted(attr_names):
        vals = []
        ok_numeric = True

        for i in range(graph.num_nodes()):
            orig_id = inv[i]
            v = G.nodes[orig_id].get(attr, 0)

            # Treat missing as 0; reject non-numeric values
            if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
                vals.append(v)
            else:
                ok_numeric = False
                break

        if not ok_numeric:
            # Skip attributes that aren't numeric 
            continue

        # Use int64 if values look like integers/bools, else float32
        all_intlike = all(float(v).is_integer() for v in vals)
        if all_intlike:
            graph.ndata[attr] = torch.tensor(vals, dtype=torch.int64)
        else:
            graph.ndata[attr] = torch.tensor(vals, dtype=torch.float32)

    return graph


def create(params):
    """
    Returns a GDL graph of given type and size.
    """
    assert isinstance(params, HyperParameters)
    # Create friendly dictionary from list of (name, function) tuples
    members = dict(inspect.getmembers(sys.modules[__name__], inspect.isfunction))
    constructor = members.get(params.kind)
    if constructor is None:
        raise Exception(f"Invalid graph type: {params.kind}")
    # Construct DGL graph
    return constructor(params=params)
