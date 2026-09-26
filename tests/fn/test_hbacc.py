"""hbacc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hbacc import hbond_acceptor_count


def test_hbacc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hbond_acceptor_count(smiles=None)
