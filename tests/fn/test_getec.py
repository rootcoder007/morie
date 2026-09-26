"""getec is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.getec import getec


def test_getec_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        getec()
