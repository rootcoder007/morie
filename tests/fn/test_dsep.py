"""Tests for dsep.d_separation."""

from morie.fn import _array_core as np

from morie.fn.dsep import d_separation


def test_dsep_basic():
    """Test basic functionality.

    d-separation is a purely graph-theoretic question: given a DAG and
    three (possibly empty) sets of node names, decide whether every path
    between x and y is blocked by z. The nodes are strings drawn from the
    DAG's vertex set; there is no data matrix involved.
    """
    # Chain A -> B -> C: B is a chain node on the unique path A --- C.
    # Conditioning on B blocks that path, so A and C are d-separated
    # given {B}.
    dag = {"A": [], "B": ["A"], "C": ["B"]}
    x = "A"
    y = "C"
    z = ("B",)

    result = d_separation(dag, x, y, z)

    # The shelf function returns a RichResult. Access its payload dict
    # for key-based assertions.
    assert hasattr(result, "payload")
    payload = result.payload

    # Documented payload keys: dseparated, npaths, nnodes.
    assert "dseparated" in payload
    assert "npaths" in payload
    assert "nnodes" in payload

    # In A -> B -> C with z = {B}, A and C are d-separated.
    assert payload["dseparated"] is True or payload["dseparated"] == True

    # npaths must be a non-negative integer count of paths examined.
    npaths = payload["npaths"]
    assert isinstance(npaths, int)
    assert npaths >= 0

    # nnodes must be a positive integer count of vertices in the DAG.
    nnodes = payload["nnodes"]
    assert isinstance(nnodes, int)
    assert nnodes == 3  # {"A", "B", "C"} has exactly three nodes


def test_dsep_edge():
    """Test edge cases.

    Same chain DAG, but now the conditioning set is empty. The single
    chain A -> B -> C is open (no collider, no conditioning node), so
    A and C are NOT d-separated given the empty set.
    """
    dag = {"A": [], "B": ["A"], "C": ["B"]}
    x = "A"
    y = "C"
    z = ()  # empty conditioning set

    result = d_separation(dag, x, y, z)

    assert hasattr(result, "payload")
    payload = result.payload

    assert "dseparated" in payload
    assert "npaths" in payload
    assert "nnodes" in payload

    # With no conditioning, A and C are connected by an open chain
    # A -> B -> C, so they are not d-separated.
    assert payload["dseparated"] is False or payload["dseparated"] == False

    npaths = payload["npaths"]
    assert isinstance(npaths, int)
    assert npaths >= 0

    nnodes = payload["nnodes"]
    assert isinstance(nnodes, int)
    assert nnodes == 3
