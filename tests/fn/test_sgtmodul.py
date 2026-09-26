"""sgtmodul is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sgtmodul import sgt_modularity_matrix


def test_sgtmodul_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sgt_modularity_matrix(A=None)
