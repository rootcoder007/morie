% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.entheo_notch}
\alias{.entheo_notch}
\title{.entheo_notch}
\usage{
.entheo_notch(x, sfreq, freq, q = 30)
}
\arguments{
\item{x}{A matrix; indexed by row and column.}
\item{sfreq}{Numeric; combined arithmetically in the body.}
\item{freq}{Numeric; combined arithmetically in the body.}
\item{q}{Numeric; combined arithmetically in the body. Defaults to \code{30}.}
}
\description{
A step of the entheo_preprocess implementation. Called by \code{preprocess_eeg}. See the file header for the source the module follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
.entheo_notch(x = matrix(c(1, 2, 3, 4, 5, 6), nrow = 2), sfreq = c(1, 2, 3, 4, 5, 6, 7, 8), freq = c(1, 2, 3, 4, 5, 6, 7, 8))
}
