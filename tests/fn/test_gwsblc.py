"""gwsblc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gwsblc import gwas_block_combine


def test_gwsblc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gwas_block_combine(block_results=None)
