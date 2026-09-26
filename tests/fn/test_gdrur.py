"""gdrur is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gdrur import gdrur


def test_gdrur_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gdrur()
