% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.logistic_loss}
\alias{.logistic_loss}
\title{.logistic_loss}
\usage{
.logistic_loss(y, phi_val)
}
\arguments{
\item{y}{See Usage.}
\item{phi_val}{See Usage.}
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
