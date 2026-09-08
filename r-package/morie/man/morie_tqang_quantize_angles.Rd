% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_tqang_quantize_angles}
\alias{morie_tqang_quantize_angles}
\alias{morie_tqang_tqang}
\alias{morie_tqang_turboquant_angle_quantization}
\title{Uniform quantisation of angles to \code{2^bits} equal sectors}
\usage{
morie_tqang_quantize_angles(theta, bits = 4L) 

morie_tqang_tqang(theta, bits = 4L) 

morie_tqang_turboquant_angle_quantization(theta, bits = 4L) 
}
\arguments{
\item{theta}{Numeric vector of angles.}
\item{bits}{Bits per symbol (1..30).}
}
\description{
Returns a list with the sector indices, the reconstructed angles, the wrapped errors, the empirical MSE, the worst-case absolute error, the sector width \code{delta}, \code{half_delta}, the theoretical MSE bound \code{Delta^2/12}, the \code{bits} and \code{levels}.
}
\value{
A list with \code{estimate}, \code{indices}, \code{values}, \code{errors}, \code{mse}, \code{max_abs_error}, \code{delta}, \code{half_delta}, \code{mse_bound}, \code{bits}, \code{levels}, \code{method}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
morie_tqang_quantize_angles(V)
}
