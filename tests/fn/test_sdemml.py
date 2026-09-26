"""sdemml is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sdemml import sdemml


def test_sdemml_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sdemml(y=None, X=None, W=None)
