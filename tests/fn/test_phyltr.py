"""phyltr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.phyltr import phylogenetic_tree_nj


def test_phyltr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        phylogenetic_tree_nj(distance_matrix=None)
