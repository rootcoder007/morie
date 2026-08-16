% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_dag_dsep}
\alias{.morie_dag_dsep}
\title{Internal helper: d-separation of x and y given z (Bayes-Ball)}
\usage{
.morie_dag_dsep(g, x, y, z)
}
\description{
Internal helper: d-separation of x and y given z (Bayes-Ball)
}
\value{
The terminal expression of the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
D <- data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9))
.morie_dag_desc(D, V)
.morie_dag_dsep(g = D, x = D, y = D, z = D)
}
