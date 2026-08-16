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
\item{Q}{See Usage.}
\item{K}{See Usage.}
\item{V}{See Usage.}
\item{n_heads}{See Usage.}
\item{n_kv_heads}{See Usage.}
\item{mask}{Defaults to \code{NULL}.}
\item{positions}{Defaults to \code{NULL}.}
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
