"""sptsmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sptsmp import sptsmp


def test_sptsmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sptsmp()
