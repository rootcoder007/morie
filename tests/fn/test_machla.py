"""machla is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.machla import machla


def test_machla_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        machla()
