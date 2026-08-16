% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_p_label_value}
\alias{.siu_p_label_value}
\title{.siu_p_label_value}
\usage{
.siu_p_label_value(text, label)
}
\arguments{
\item{text}{See Usage.}
\item{label}{Passed to \code{.siu_p_re_escape}.}
}
\description{
Label / value primitives. ---------------------------------------------------------------------------
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.siu_p_label_value(V, V)
}
