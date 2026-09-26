"""ghgap is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghgap import ghgap


def test_ghgap_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghgap()
