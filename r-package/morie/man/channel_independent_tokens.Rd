% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{channel_independent_tokens}
\alias{channel_independent_tokens}
\title{One token stream per channel, weights shared across channels}
\usage{
channel_independent_tokens(X, patch_len, stride = NULL)
}
\arguments{
\item{X}{See Usage.}
\item{patch_len}{See Usage.}
\item{stride}{Defaults to \code{NULL}.}
}
\description{
Nothing is mixed across channels, so the channel identity survives -- and because the downstream weights are shared, the whole model is equivariant to permuting them.
}
\value{
A list with \code{tokens}, \code{D}, \code{n_patches}, \code{patch_len}, \code{n_tokens_total}, \code{design}, \code{note}.
}
\keyword{internal}
\examples{
channel_independent_tokens(X = c(1, 2, 3, 4, 5, 6, 7, 8), patch_len = 5L)
}
