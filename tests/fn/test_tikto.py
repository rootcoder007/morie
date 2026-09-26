"""tikto is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tikto import tiktoken_bpe


def test_tikto_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tiktoken_bpe(corpus=None)
