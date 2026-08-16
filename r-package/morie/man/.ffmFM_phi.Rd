% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.ffmFM_phi}
\alias{.ffmFM_phi}
\title{.ffmFM_phi}
\usage{
.ffmFM_phi(x, fields, W)
}
\arguments{
\item{x}{See Usage.}
\item{fields}{See Usage.}
\item{W}{See Usage.}
}
\description{
Part of the ffmFM_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{tot}, as built in the body.
}
\references{
Juan, Y., Zhuang, Y., Chin, W.-S. & Lin, C.-J. (2016) "Field-aware
Factorization Machines for CTR Prediction", Proceedings of the Tenth
ACM Conference on Recommender Systems (RecSys \\'16), 43-50,
doi:10.1145/2959100.2959134. Sec. 2 (FM and its limitation), Sec. 3
(the FFM model equation with the crossed field indices, the nfk
parameter count, the logistic loss with y in {-1,1}, and the AdaGrad
updates of eqs. (8)-(9)), and Sec. 3.3 (overfitting and early
stopping).
Rendle, S. (2010) "Factorization Machines", ICDM 2010, 995-1000,
doi:10.1109/ICDM.2010.127. The model FFM specialises.
}
\keyword{internal}
