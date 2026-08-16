% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.entheo_asr_trim}
\alias{.entheo_asr_trim}
\title{.entheo_asr_trim}
\usage{
.entheo_asr_trim(x, threshold)
}
\arguments{
\item{x}{A vector; indexed elementwise.}
\item{threshold}{See Usage.}
}
\description{
A step of the entheo_preprocess implementation. Called by \code{preprocess_eeg}. See the file header for the source the module follows.
}
\value{
A list with \code{arr}, \code{n_bad}.
}
\keyword{internal}
\examples{
M <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2)
.entheo_asr_trim(M, M)
}
