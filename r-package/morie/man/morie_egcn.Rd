% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_egcn}
\alias{morie_egcn}
\title{Re-export from egnnL}
\usage{
morie_egcn(H, X, layers, phi_e, phi_x, phi_h, A = NULL, C = NULL)
}
\arguments{
\item{H}{Passed to \code{morie_egnnL}.}
\item{X}{Passed to \code{morie_egnnL}.}
\item{layers}{Passed to \code{morie_egnnL}.}
\item{phi_e}{Passed to \code{morie_egnnL}.}
\item{phi_x}{Passed to \code{morie_egnnL}.}
\item{phi_h}{Passed to \code{morie_egnnL}.}
\item{A}{Passed to \code{morie_egnnL}.}
\item{C}{Passed to \code{morie_egnnL}.}
}
\description{
A step of the egcn_native implementation. No other function in the package calls it. See the file header for the source the module follows.
}
\value{
The value of \code{morie_egnnL}.
}
\references{
Satorras, V. G., Hoogeboom, E. & Welling, M. (2021) "E(n) Equivariant
Graph Neural Networks", Proceedings of the 38th International
Conference on Machine Learning (ICML 2021), PMLR 139, 9323-9332,
arXiv:2102.09844. Sec. 3 (the EGCL of eqs. (3)-(6), with C = 1/(M-1)).
Sec. 3.1 (the equivariance condition Qx + g). Sec. 3.2 (the momentum
variant replacing eq. (4)).
Thomas, N., Smidt, T., Kearnes, S., Yang, L., Li, L., Kohlhoff, K. &
Riley, P. (2018) "Tensor Field Networks", arXiv:1802.08219.
}
\keyword{internal}
