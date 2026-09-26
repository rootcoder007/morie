"""secli is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.secli import secli


def test_secli_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        secli()
