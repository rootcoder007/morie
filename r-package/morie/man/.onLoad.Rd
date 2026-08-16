% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.onLoad}
\alias{.onLoad}
\title{Future\\'s connection-misuse check (diff_connections() in}
\usage{
.onLoad(libname, pkgname)
}
\arguments{
\item{libname}{See Usage.}
\item{pkgname}{See Usage.}
}
\description{
FutureResult) can segfault R uncatchably when DoubleML/mlr3 resolve futures. Setting the env var BEFORE future is loaded makes the "ignore" setting take effect in the main process AND every worker -- each re-reads R_FUTURE_* when future loads. An options() guard does not reach workers, which is why it was only flaky. Only set when the user has not chosen their own value.
}
\value{
Invisibly,nothing; the function is called for its effect.
}
\keyword{internal}
