"""sr2sl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sr2sl import sr2sl


def test_sr2sl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sr2sl()
