"""saldi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.saldi import saldi


def test_saldi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        saldi()
