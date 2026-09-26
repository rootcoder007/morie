"""svcli is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svcli import cut_line


def test_svcli_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cut_line(data=None)
