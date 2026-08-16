% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.sv_dl_components}
\alias{.sv_dl_components}
\title{Connected components (union-find); singletons dropped}
\usage{
.sv_dl_components(n, edges)
}
\arguments{
\item{n}{A count; the body uses it as \code{seq_len(...)}.}
\item{edges}{A matrix; indexed by row and column.}
}
\description{
Part of the sv_dl_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
