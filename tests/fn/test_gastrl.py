"""gastrl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gastrl import gastrl


def test_gastrl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gastrl()
