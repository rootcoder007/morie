% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mistr_mistral_block}
\alias{mistr_mistral_block}
\title{mistr_mistral_block}
\usage{
mistr_mistral_block(X, Wq, Wk, Wv, Wo, W1, W2, W3, n_heads, n_kv_heads, window, norm1 = NULL, 
    norm2 = NULL, base = 10000) 
}
\arguments{
\item{X}{A matrix; passed to \code{as.matrix}.}
\item{Wq}{Passed to \code{proj}.}
\item{Wk}{Passed to \code{proj}.}
\item{Wv}{Passed to \code{proj}.}
\item{Wo}{Passed to \code{proj}.}
\item{W1}{Passed to \code{mistr_swiglu}.}
\item{W2}{Passed to \code{mistr_swiglu}.}
\item{W3}{Passed to \code{mistr_swiglu}.}
\item{n_heads}{Carried through into a list the body builds.}
\item{n_kv_heads}{Numeric; combined arithmetically in the body.}
\item{window}{Coerced to integer by the body, with \code{as.integer}.}
\item{norm1}{Optional; may be \code{NULL}. Passed to \code{mistr_rms_norm}.}
\item{norm2}{Passed to \code{mistr_rms_norm}.}
\item{base}{Passed to \code{mistr_grouped_query_attention}. Defaults to \code{10000}.}
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
\examples{
set.seed(1)
d <- 8; nh <- 4; nkv <- 2; hd <- 2; dff <- 16; L <- 4
X <- matrix(rnorm(L * d), L, d)
Wq <- matrix(rnorm(d * d, 0, 0.2), d, d)
Wk <- matrix(rnorm(d * nkv * hd, 0, 0.2), d, nkv * hd)
Wv <- matrix(rnorm(d * nkv * hd, 0, 0.2), d, nkv * hd)
Wo <- matrix(rnorm(d * d, 0, 0.2), d, d)
W1 <- matrix(rnorm(d * dff, 0, 0.2), d, dff)
W3 <- matrix(rnorm(d * dff, 0, 0.2), d, dff)
W2 <- matrix(rnorm(dff * d, 0, 0.2), dff, d)
r <- mistr_mistral_block(X, Wq, Wk, Wv, Wo, W1, W2, W3, nh, nkv,
                         window = 3)
dim(r$output)
}
