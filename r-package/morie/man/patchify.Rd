% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{patchify}
\alias{patchify}
\title{Cut one univariate series into patches of length P. With stride}
\usage{
patchify(x, patch_len, stride = NULL)
}
\arguments{
\item{x}{See Usage.}
\item{patch_len}{See Usage.}
\item{stride}{Defaults to \code{NULL}.}
}
\description{
equal to patch_len the patches are disjoint; a smaller stride overlaps them. The number of patches is N = floor((L-P)/S) + 1, which is what makes the attention map shrink.
}
\value{
A list with \code{patches}, \code{n_patches}, \code{patch_len}, \code{stride}, \code{L}, \code{covers}.
}
\keyword{internal}
