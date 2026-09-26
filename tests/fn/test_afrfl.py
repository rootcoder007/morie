"""afrfl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afrfl import afrfl


def test_afrfl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afrfl()
