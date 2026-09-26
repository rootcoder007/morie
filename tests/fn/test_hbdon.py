"""hbdon is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hbdon import hbond_donor_count


def test_hbdon_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hbond_donor_count(smiles=None)
