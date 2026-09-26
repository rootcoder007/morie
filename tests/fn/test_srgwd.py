"""srgwd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srgwd import srgwd


def test_srgwd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srgwd()
