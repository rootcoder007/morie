"""hydfl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hydfl import hydfl


def test_hydfl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hydfl()
