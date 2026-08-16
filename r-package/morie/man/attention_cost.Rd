% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{attention_cost}
\alias{attention_cost}
\title{Attention-map size, patched against point-wise. Point-wise}
\usage{
attention_cost(L, patch_len, stride = NULL, D = 1, channel_independent = TRUE)
}
\arguments{
\item{L}{See Usage.}
\item{patch_len}{See Usage.}
\item{stride}{Defaults to \code{NULL}.}
\item{D}{Defaults to \code{1}.}
\item{channel_independent}{Defaults to \code{TRUE}.}
}
\description{
attention over a look-back of L costs L^2 per channel. Patching reduces the sequence to N tokens, so the cost falls to N^2.
}
\value{
A list with \code{n_patches}, \code{pointwise}, \code{patched}, \code{reduction}, \code{stride}, \code{patch_len}, \code{note}.
}
\keyword{internal}
\examples{
attention_cost(L = 5L, patch_len = 5L)
}
