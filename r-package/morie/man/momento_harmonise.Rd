% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{momento_harmonise}
\alias{momento_harmonise}
\title{momento_harmonise}
\usage{
momento_harmonise(series_list, patch_len, normalise = TRUE)
}
\arguments{
\item{series_list}{A vector; its length is taken.}
\item{patch_len}{Coerced to integer by the body, with \code{as.integer}.}
\item{normalise}{A flag; the body branches on it. Defaults to \code{TRUE}.}
}
\description{
Part of the momento_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{batch}, \code{meta}, \code{n_series}, \code{n_patches}, \code{patch_len}, \code{note}.
}
\references{
  Goswami et al. (2024) "MOMENT" arXiv:2402.03885
  Devlin et al. (2019) "BERT" arXiv:1810.04805
  Nie et al. (2023) "PatchTST" arXiv:2211.14730
Base R only.
}
\keyword{internal}
