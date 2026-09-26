"""hydra is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hydra import hydra


def test_hydra_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hydra()
