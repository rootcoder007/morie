"""dlauae is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dlauae import dlauae


def test_dlauae_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dlauae()
