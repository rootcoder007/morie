% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.arsau_load_one}
\alias{.arsau_load_one}
\title{.arsau_load_one}
\usage{
.arsau_load_one(entry, data_dir = NULL, language = "en", allow_invalid = FALSE)
}
\arguments{
\item{entry}{A list; the body reads \code{$csv_filename}, \code{$description_en}, \code{$description_fr}, \code{$expected_cols}, \code{$expected_rows}, \code{$is_valid}, \code{$kind}, \code{$sidecar_filename}, \code{$year_or_range} from it.}
\item{data_dir}{Passed to \code{.morie_resolve_arsau_dir}.}
\item{language}{Character; passed to \code{substr}. Defaults to \code{"en"}.}
\item{allow_invalid}{A flag; the body branches on it. Defaults to \code{FALSE}.}
}
\description{
A step of the arsau implementation. Called by \code{morie_arsau_load_aggregate_summary}, \code{morie_arsau_load_detailed_dataset}, \code{morie_arsau_load_individual_records} and 3 others in the module. See the file header for the source the module follows.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
