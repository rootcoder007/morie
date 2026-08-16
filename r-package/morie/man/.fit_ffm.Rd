% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.fit_ffm}
\alias{.fit_ffm}
\title{.fit_ffm}
\usage{
.fit_ffm(rows, labels, fields, n_features, n_fields, k_dim = 4L, eta = 0.1, lam = 2e-05, epochs = 10L, seed = 0L)
}
\arguments{
\item{rows}{A vector; its length is taken and its elements indexed.}
\item{labels}{A vector; its length is taken and its elements indexed.}
\item{fields}{A vector; indexed elementwise.}
\item{n_features}{Coerced to integer by the body, with \code{as.integer}.}
\item{n_fields}{Coerced to integer by the body, with \code{as.integer}.}
\item{k_dim}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{4L}.}
\item{eta}{Numeric; combined arithmetically in the body. Defaults to \code{0.1}.}
\item{lam}{Numeric; combined arithmetically in the body. Defaults to \code{2e-05}.}
\item{epochs}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{10L}.}
\item{seed}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{0L}.}
}
\description{
A step of the ffmFM_native implementation. Called by \code{morie_ffmFM}. See the file header for the source the module follows.
}
\value{
A list with \code{estimate}, \code{W}, \code{loss_history}, \code{final_loss}, \code{k}, \code{n_parameters}, \code{n_parameters_fm}, \code{method}, \code{caveat}.
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
