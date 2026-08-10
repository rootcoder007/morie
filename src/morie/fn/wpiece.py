# morie.fn -- wave 3 slice w5_00 (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""WordPiece tokenizer -- likelihood-driven subword merges.

Same method as :mod:`morie.fn.hmwpt`: WordPiece trains by merging the
symbol pair that most increases corpus unigram likelihood -- after
cancelling constants, the pair maximising
score(A, B) = freq(AB) / (freq(A) freq(B)) -- with ## continuation
prefixes and greedy longest-match-first segmentation. Described by
Schuster, M. and Nakajima, K. (2012), "Japanese and Korean voice
search", IEEE ICASSP 2012, pp. 5149-5152 (the original WordPiece);
the merge criterion and ## convention as used here follow the BERT
lineage: Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019),
"BERT: Pre-training of Deep Bidirectional Transformers for Language
Understanding", NAACL-HLT 2019, arXiv:1810.04805 (Section 4.1 /
WordPiece vocabulary), and Wu, Y. et al. (2016), "Google's Neural
Machine Translation System", arXiv:1609.08144 (Section 4.1 wordpiece
model, likelihood-maximising segmentation).

There is exactly one implementation: this module delegates to
:func:`morie.fn.hmwpt.geron_wordpiece_tokenizer`.

Sources: fetched-wave3/devlin-etal-2019-bert-arxiv1810.04805.pdf,
fetched-wave3/wu-etal-2016-gnmt-wordpiece-arxiv1609.08144.pdf
(Schuster-Nakajima 2012 is IEEE-paywalled; the GNMT paper Sec 4.1
restates the likelihood criterion implemented here).
"""

from .hmwpt import geron_wordpiece_tokenizer as _impl

__all__ = ["wpiece", "wordpiece"]


def wpiece(corpus, vocab_size=50):
    """WordPiece tokenizer (Schuster-Nakajima 2012; Wu et al. 2016 Sec 4.1).

    Delegates to :func:`morie.fn.hmwpt.geron_wordpiece_tokenizer`; see
    that function for parameters and payload.
    """
    return _impl(corpus, vocab_size=vocab_size)


wordpiece = wpiece


def cheatsheet():
    return "wpiece: WordPiece tokenizer (Schuster-Nakajima 2012; Wu et al. 2016) -- alias of hmwpt.geron_wordpiece_tokenizer"
