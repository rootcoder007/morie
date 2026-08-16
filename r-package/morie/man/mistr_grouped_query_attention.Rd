% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mistr_grouped_query_attention}
\alias{mistr_grouped_query_attention}
\title{mistr_grouped_query_attention}
\usage{
mistr_grouped_query_attention(Q, K, V, n_heads, n_kv_heads, mask = NULL, positions = NULL, base = 10000)
}
\arguments{
\item{Q}{A matrix; passed to \code{as.matrix}.}
\item{K}{A matrix; passed to \code{as.matrix}.}
\item{V}{A matrix; passed to \code{as.matrix}.}
\item{n_heads}{A count; the body uses it as \code{seq_len(...)}.}
\item{n_kv_heads}{Numeric; combined arithmetically in the body.}
\item{mask}{Optional; may be \code{NULL}. A matrix; indexed by row and column.}
\item{positions}{Optional; may be \code{NULL}. A vector; its length is taken.}
\item{base}{Defaults to \code{10000}.}
}
\description{
Part of the mistr_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
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
