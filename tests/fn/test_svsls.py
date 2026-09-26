"""svsls is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svsls import salience_model


def test_svsls_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        salience_model(data=None)
