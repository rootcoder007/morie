"""oralb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.oralb import oral_bioavailability


def test_oralb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        oral_bioavailability(smiles=None)
