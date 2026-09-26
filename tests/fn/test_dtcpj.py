"""dtcpj is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtcpj import dtcpj


def test_dtcpj_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtcpj()
