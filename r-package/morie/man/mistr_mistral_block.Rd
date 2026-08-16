% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mistr_mistral_block}
\alias{mistr_mistral_block}
\title{mistr_mistral_block}
\usage{
mistr_mistral_block(X, Wq, Wk, Wv, Wo, W1, W2, W3, n_heads, n_kv_heads, window, norm1 = NULL, norm2 = NULL, base = 10000)
}
\arguments{
\item{X}{A matrix; passed to \code{as.matrix}.}
\item{Wq}{See Usage.}
\item{Wk}{See Usage.}
\item{Wv}{See Usage.}
\item{Wo}{See Usage.}
\item{W1}{See Usage.}
\item{W2}{See Usage.}
\item{W3}{See Usage.}
\item{n_heads}{See Usage.}
\item{n_kv_heads}{Numeric; combined arithmetically in the body.}
\item{window}{See Usage.}
\item{norm1}{Defaults to \code{NULL}.}
\item{norm2}{Defaults to \code{NULL}.}
\item{base}{Defaults to \code{10000}.}
}
\description{
Part of the mistr_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{output}, \code{attention_mask}, \code{L}, \code{d}, \code{n_heads}, \code{n_kv_heads}, \code{window}, \code{kv_cache_entries}, \code{method}.
}
\references{
  Jiang et al. (2023) "Mistral 7B" arXiv:2310.06825
  Su et al. (2024) "RoFormer" Neurocomputing 568, 127063
  Shazeer (2020) "GLU Variants" arXiv:2002.05202
  Zhang & Sennrich (2019) "RMSNorm" arXiv:1910.07467
  Ainslie et al. (2023) "GQA" arXiv:2305.13245
  Beltagy et al. (2020) "Longformer" arXiv:2004.05150
Base R only.
}
\keyword{internal}
