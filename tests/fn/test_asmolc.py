"""asmolc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.asmolc import olc_assembly


def test_asmolc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        olc_assembly(long_reads=None)
