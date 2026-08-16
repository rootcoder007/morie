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
\item{H}{See Usage.}
\item{X}{See Usage.}
\item{layers}{See Usage.}
\item{phi_e}{See Usage.}
\item{phi_x}{See Usage.}
\item{phi_h}{See Usage.}
\item{A}{Defaults to \code{NULL}.}
\item{C}{Defaults to \code{NULL}.}
}
\description{
Part of the egcn_native implementation; see the file header for the source it follows.
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
