"""protdis is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.protdis import protein_disorder


def test_protdis_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        protein_disorder(sequence=None)
