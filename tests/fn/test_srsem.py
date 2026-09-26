"""srsem is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srsem import srsem


def test_srsem_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srsem()
