"""micrR is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.micrR import microsoft_sr


def test_micrR_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        microsoft_sr(x=None)
