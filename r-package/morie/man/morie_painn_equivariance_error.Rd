% Generated from the module source by ledger/gen_rd.R -- roxygen2 hangs
% on this package. Usage comes from args(), so it tracks the formals.
\name{morie_painn_equivariance_error}
\alias{morie_painn_equivariance_error}
\title{Morie.fn -- function file (rootcoder007/morie)}
\usage{
morie_painn_equivariance_error(model, s, v, R, Q, tol = 1e-09)
}
\description{
morie.fn -- function file (rootcoder007/morie) PaiNN: equivariant message passing for tensorial properties. "Equivariant message passing for the prediction of tensorial properties and molecular spectra", ICML 2021, PMLR 139, 9377-9388, arXiv:2102.03150. Message passing networks scale readily to large training sets but have proven less data efficient than kernel methods; the identification of the limitations of invariant representations as a major reason; the extension of message passing to rotationally equivariant representations; the polarizable atom interaction neural network improving on common molecule benchmarks while reducing model size and inference time; and the use of equivariant atomwise representations for tensorial properties and molecular spectra, with speedups of 4-5 orders of magnitude over the electronic structure reference. Schutt, K. T., Kindermans, P.-J., Sauceda, H. E., Chmiela, S., arXiv:1706.08566. The invariant predecessor. Graph Neural Networks", ICML 2021, PMLR 139, 9323-9332, arXiv:2102.09844.
}
\value{
A list with components \code{nrow}, \code{ncol}, \code{scalar_error}, \code{vector_error}, \code{scalars_invariant}, \code{vectors_equivariant}, \code{note}, \code{sep}.
}
\references{
Sources: Schutt, K. T., Unke, O. T. & Gastegger, M. (2021)
Tkatchenko, A. & Muller, K.-R. (2017) "SchNet", NeurIPS 2017,
Satorras, V. G., Hoogeboom, E. & Welling, M. (2021) "E(n) Equivariant
}
\keyword{internal}
