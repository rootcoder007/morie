% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_offlrl}
\alias{morie_offlrl}
\title{morie_offlrl}
\usage{
morie_offlrl(dataset, states = NULL, actions = NULL, alpha = 1, gamma = 0.99, variant = "H", backup = "max", policy = NULL, mu = NULL, lr = 0.5, iters = 2000, tol = 1e-12)
}
\arguments{
\item{dataset}{The body requires: offlrl: dataset must be non-empty.}
\item{states}{Optional; may be \code{NULL}. Coerced to list by the body, with \code{as.list}.}
\item{actions}{Optional; may be \code{NULL}. Coerced to list by the body, with \code{as.list}.}
\item{alpha}{Numeric; combined arithmetically in the body. Defaults to \code{1}.}
\item{gamma}{Numeric; combined arithmetically in the body. Defaults to \code{0.99}.}
\item{variant}{One of \code{"H"}, \code{"mu"}, \code{"rho"}. Defaults to \code{"H"}.}
\item{backup}{One of \code{"max"}, \code{"pi"}. Defaults to \code{"max"}.}
\item{policy}{The body requires: offlrl: variant='rho' needs policy(a|s) to play the role of pi^\{k-1\}.}
\item{mu}{The body requires: offlrl: variant must be one of c('H','rho','mu'), got '.}
\item{lr}{Numeric; combined arithmetically in the body. Defaults to \code{0.5}.}
\item{iters}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{2000}.}
\item{tol}{Passed to \code{<}. Defaults to \code{1e-12}.}
}
\description{
Part of the offlrl_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{q}, \code{value}, \code{greedy}, \code{behavior}, \code{counts}, \code{penalty}, \code{bellman_error}, \code{objective}, \code{alpha}, \code{variant}, \code{backup}, \code{n_transitions}, \code{method}.
}
\keyword{internal}
\examples{
D <- data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9))
morie_offlrl(D)
}
