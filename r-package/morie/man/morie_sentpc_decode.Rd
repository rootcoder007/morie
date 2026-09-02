% Generated from the module source by ledger/gen_rd.R -- roxygen2 hangs
% on this package. Usage comes from args(), so it tracks the formals.
\name{morie_sentpc_decode}
\alias{morie_sentpc_decode}
\title{SentencePiece: subword tokenisation that is losslessly reversible}
\usage{
morie_sentpc_decode(pieces, strip_prefix = TRUE) 
}
\description{
SentencePiece: subword tokenisation that is losslessly reversible. and language independent subword tokenizer and detokenizer for Neural Text Processing", EMNLP 2018 System Demonstrations, 66-71, whitespace, BPE and unigram, fixed vocabulary size); Sennrich, R., Native R arm mirroring the Python arm exactly: the same U+2581 escaping and the same _units split that keeps every unit starting at the marker so runs of spaces do not collapse, the same greedy BPE trainer that merges the most frequent pair until the vocabulary is full, the same Viterbi over the split lattice for the unigram model, and morie_sentpc_decode/unescape as a pure string operation.
}
\arguments{
\item{pieces}{Argument \code{pieces}; see Usage.}
\item{strip_prefix}{Argument \code{strip_prefix}; see Usage.}
}
\value{
A list with components \code{collapse}.
}
\references{
Sources: Kudo, T. & Richardson, J. (2018) "SentencePiece: A simple
doi:10.18653/v1/D18-2012 (lossless encoding with U+2581 for
Haddow, B. & Birch, A. (2016) ACL 1715-1725 (BPE merges); Kudo, T.
(2018) ACL 66-75 (unigram LM and Viterbi).
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
morie_sentpc_decode(V)
}
