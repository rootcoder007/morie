% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{forecast}
\alias{forecast}
\title{forecast}
\usage{
forecast(fitted, h = 12) 
}
\arguments{
\item{fitted}{A list; the body reads \code{$ar}, \code{$ma}, \code{$order}, \code{$residuals}, \code{$s}, \code{$seasonal_order}, \code{$sigma2}, \code{$y} from it.}
\item{h}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{12}.}
}
\description{
Part of the sarima_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{forecast}, \code{variance}, \code{se}, \code{psi}, \code{method}.
}
\keyword{internal}
\examples{
set.seed(1)
y <- as.numeric(10 + sin(2 * pi * (1:72) / 12) + rnorm(72, 0, 0.3))
fit <- morie_sarima(y, order = c(0, 1, 1), seasonal_order = c(0, 1, 1),
                    s = 12)
fc <- forecast(fit, h = 12)
length(fc$mean \%||\% fc$forecast \%||\% fc[[1]]) == 12L
}
