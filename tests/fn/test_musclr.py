"""musclr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.musclr import muscle_msa


def test_musclr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        muscle_msa(sequences=None)
