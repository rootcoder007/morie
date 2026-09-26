"""rnafld is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rnafld import rna_fold


def test_rnafld_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rna_fold(sequence=None)
