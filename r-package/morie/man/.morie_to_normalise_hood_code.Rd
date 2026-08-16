% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_to_normalise_hood_code}
\alias{.morie_to_normalise_hood_code}
\title{Normalise a hood-code value to the 3-char zero-padded canonical form}
\usage{
.morie_to_normalise_hood_code(x)
}
\arguments{
\item{x}{Coerced to character by the body, with \code{as.character}.}
}
\description{
the crosswalk uses ("82" -> "082"; "0082" -> "082"; "Niagara" -> NA).
}
\value{
The value of \code{ifelse}.
}
\keyword{internal}
