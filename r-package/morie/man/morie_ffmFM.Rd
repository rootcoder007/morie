% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_ffmFM}
\alias{morie_ffmFM}
\title{morie_ffmFM}
\usage{
morie_ffmFM(rows, labels, fields, n_features, n_fields, k_dim = 4L, eta = 0.1, lam = 2e-05, epochs = 10L, seed = 0L)
}
\arguments{
\item{rows}{Passed to \code{.fit_ffm}.}
\item{labels}{Passed to \code{.fit_ffm}.}
\item{fields}{Passed to \code{.fit_ffm}.}
\item{n_features}{Passed to \code{.fit_ffm}.}
\item{n_fields}{Passed to \code{.fit_ffm}.}
\item{k_dim}{Passed to \code{.fit_ffm}. Defaults to \code{4L}.}
\item{eta}{Passed to \code{.fit_ffm}. Defaults to \code{0.1}.}
\item{lam}{Passed to \code{.fit_ffm}. Defaults to \code{2e-05}.}
\item{epochs}{Passed to \code{.fit_ffm}. Defaults to \code{10L}.}
\item{seed}{Passed to \code{.fit_ffm}. Defaults to \code{0L}.}
}
\description{
A step of the ffmFM_native implementation. No other function in the package calls it. See the file header for the source the module follows.
}
\value{
The value of \code{.fit_ffm}.
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
