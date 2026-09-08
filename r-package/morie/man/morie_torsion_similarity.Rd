% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_torsion_similarity}
\alias{morie_torsion_similarity}
\title{The paper\'s similarity score S = 2 D / (d_i + d_j)}
\usage{
morie_torsion_similarity(t1, t2) 
}
\arguments{
\item{t1,t2}{Either torsion dictionaries or iterables of codes.}
}
\description{
The paper\'s similarity score S = 2 D / (d_i + d_j)
}
\value{
A numeric similarity in \code{[0, 1]}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
morie_torsion_similarity(V, V)
}
