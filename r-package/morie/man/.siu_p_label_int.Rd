% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_p_label_int}
\alias{.siu_p_label_int}
\title{.siu_p_label_int}
\usage{
.siu_p_label_int(text, label)
}
\arguments{
\item{text}{Passed to \code{.siu_p_label_value}.}
\item{label}{Passed to \code{.siu_p_label_value}.}
}
\description{
Part of the siu_parser implementation; see the file header for the source it follows.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.siu_p_label_int(V, V)
}
