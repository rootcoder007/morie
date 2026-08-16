% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{momento_task_mask}
\alias{momento_task_mask}
\title{momento_task_mask}
\usage{
momento_task_mask(n_patches, task = "forecast", span = 1, start = NULL)
}
\arguments{
\item{n_patches}{See Usage.}
\item{task}{Defaults to \code{"forecast"}.}
\item{span}{Defaults to \code{1}.}
\item{start}{Defaults to \code{NULL}.}
}
\description{
Part of the momento_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{seq.int}.
}
\references{
  Goswami et al. (2024) "MOMENT" arXiv:2402.03885
  Devlin et al. (2019) "BERT" arXiv:1810.04805
  Nie et al. (2023) "PatchTST" arXiv:2211.14730
Base R only.
}
\keyword{internal}
