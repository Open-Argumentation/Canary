from dataclasses import dataclass


@dataclass
class Nodeset:

    def __init__(self) -> None:
        self.id = None
        self.text = None
        self.nodes = []
        self.edges = []
        self.locutions = []
        super().__init__()

    def __repr__(self) -> str:
        return self.id

    @property
    def contains_schemes(self) -> bool:
        if self.nodes != [] and self.nodes is not None:
            for node in self.nodes:
                if node.scheme is not None:
                    return True
        return False

    def get_scheme_nodes(self) -> list:
        nodes = []
        if self.nodes != [] and self.nodes is not None:
            for node in self.nodes:
                if node.scheme is not None:
                    nodes.append(node)
        return nodes

    def load(self, file) -> None:
        if len(file["nodes"]) == 0:
            self.nodes = None
        else:
            for n in file["nodes"]:
                scheme = None
                scheme_id = None
                if "scheme" in n:
                    scheme = n['scheme']
                if 'schemeID' in n:
                    scheme_id = n['schemeID']
                self.nodes.append(Node(n["nodeID"], n["text"], n["type"], n["timestamp"], scheme, scheme_id))
        if len(file["edges"]) == 0:
            self.edges = None
        else:
            for e in file["edges"]:
                self.edges.append(Edge(e["edgeID"], e["fromID"], e["toID"], e["formEdgeID"]))
        if len(file["locutions"]) == 0:
            self.locutions = None
        else:
            for l in file["locutions"]:
                self.locutions.append(
                    Locution(l["nodeID"], l["personID"], l["timestamp"], l["start"],
                             l["end"],
                             l["source"]))


@dataclass
class Edge:

    def __init__(self, edge_id: str, from_id: str, to_id: str, form_edge_id: str) -> None:
        self.edgeID: str = edge_id
        self.from_id = from_id
        self.to_id = to_id
        self.form_edge_id = form_edge_id
        super().__init__()

    def __repr__(self) -> str:
        return f"{self.from_id} -> {self.to_id}"


@dataclass
class Locution:

    def __init__(self, node_id, person_id, timestamp, start, end, source) -> None:
        self.nodeID = node_id
        self.person_id = person_id
        self.timestamp = timestamp
        self.start = start
        self.end = end
        self.source = source
        super().__init__()

    def __repr__(self) -> str:
        return f"{self.start} -> {self.end}"


@dataclass
class Node:

    def __init__(self, node_id, text, node_type, timestamp, scheme, scheme_id) -> None:
        self.node_id = node_id
        self.text = text
        self.node_type = node_type
        self.scheme = scheme
        self.scheme_id = scheme_id
        self.timestamp = timestamp
        super().__init__()

    def __repr__(self) -> str:
        if self.scheme is None:
            return f"{self.node_id}: {self.text}"
        else:
            return f"{self.node_id}: {self.scheme}"
