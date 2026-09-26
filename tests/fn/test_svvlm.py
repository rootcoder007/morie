"""svvlm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svvlm import valence_model


def test_svvlm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        valence_model(data=None)
