% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.arsau_make_entry}
\alias{.arsau_make_entry}
\title{.arsau_make_entry}
\usage{
.arsau_make_entry(year_or_range, kind, csv_filename, sidecar_filename, expected_rows, expected_cols, is_valid, description_en, description_fr)
}
\arguments{
\item{year_or_range}{Carried through into a list the body builds.}
\item{kind}{Carried through into a list the body builds.}
\item{csv_filename}{Carried through into a list the body builds.}
\item{sidecar_filename}{Carried through into a list the body builds.}
\item{expected_rows}{Coerced to integer by the body, with \code{as.integer}.}
\item{expected_cols}{Coerced to integer by the body, with \code{as.integer}.}
\item{is_valid}{A flag; the body branches on it.}
\item{description_en}{Carried through into a list the body builds.}
\item{description_fr}{Carried through into a list the body builds.}
}
\description{
A step of the arsau implementation. No other function in the package calls it. See the file header for the source the module follows.
}
\value{
A list with \code{year_or_range}, \code{kind}, \code{csv_filename}, \code{sidecar_filename}, \code{expected_rows}, \code{expected_cols}, \code{is_valid}, \code{description_en}, \code{description_fr}.
}
\keyword{internal}
