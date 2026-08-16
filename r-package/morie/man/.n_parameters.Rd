% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.n_parameters}
\alias{.n_parameters}
\title{.n_parameters}
\usage{
.n_parameters(n_features, n_fields, k_dim, model = "ffm")
}
\arguments{
\item{n_features}{Coerced to integer by the body, with \code{as.integer}.}
\item{n_fields}{Coerced to integer by the body, with \code{as.integer}.}
\item{k_dim}{Coerced to integer by the body, with \code{as.integer}.}
\item{model}{One of \code{"ffm"}, \code{"fm"}. Defaults to \code{"ffm"}.}
}
\description{
A step of the ffmFM_native implementation. Called by \code{.fit_ffm}. See the file header for the source the module follows.
}
\value{
One of two values, depending on the branch taken.
}
\references{
Juan, Y., Zhuang, Y., Chin, W.-S. & Lin, C.-J. (2016) "Field-aware
Factorization Machines for CTR Prediction", Proceedings of the Tenth
ACM Conference on Recommender Systems (RecSys \'16), 43-50,
doi:10.1145/2959100.2959134. Sec. 2 (FM and its limitation), Sec. 3
(the FFM model equation with the crossed field indices, the nfk
parameter count, the logistic loss with y in {-1,1}, and the AdaGrad
updates of eqs. (8)-(9)), and Sec. 3.3 (overfitting and early
stopping).
Rendle, S. (2010) "Factorization Machines", ICDM 2010, 995-1000,
doi:10.1109/ICDM.2010.127. The model FFM specialises.
}
\keyword{internal}
\examples{
.n_parameters(n_features = c(1, 2, 3, 4, 5, 6, 7, 8), n_fields = c(1, 2, 3, 4, 5, 6, 7, 8), k_dim = c(1, 2, 3, 4, 5, 6, 7, 8))
}
