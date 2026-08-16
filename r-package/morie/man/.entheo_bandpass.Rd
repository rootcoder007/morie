% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.entheo_bandpass}
\alias{.entheo_bandpass}
\title{.entheo_bandpass}
\usage{
.entheo_bandpass(x, sfreq, low, high, order = 4L)
}
\arguments{
\item{x}{A matrix; indexed by row and column.}
\item{sfreq}{Numeric; combined arithmetically in the body.}
\item{low}{Numeric; combined arithmetically in the body.}
\item{high}{Numeric; combined arithmetically in the body.}
\item{order}{Passed to \code{.morie_dsp_butter}. Defaults to \code{4L}.}
}
\description{
A step of the entheo_preprocess implementation. Called by \code{preprocess_eeg}. See the file header for the source the module follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
