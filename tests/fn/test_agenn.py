"""agenn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agenn import agenn


def test_agenn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agenn()
