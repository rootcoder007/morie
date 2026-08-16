% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.tps_sp_project_xy}
\alias{.tps_sp_project_xy}
\title{Cosine-corrected planar projection on the Toronto bbox. The Python}
\usage{
.tps_sp_project_xy(lat, lon, lat_ref = (43.55 + 43.9)/2, lon_ref = (-79.65 + -79.1)/2)
}
\arguments{
\item{lat}{See Usage.}
\item{lon}{See Usage.}
\item{lat_ref}{Defaults to \code{(43.55 + 43.9)/2}.}
\item{lon_ref}{Defaults to \code{(-79.65 + -79.1)/2}.}
}
\description{
port delegates to morie.tps_render.project_xy; in R we approximate with a midpoint cos-lat factor so this module does not hard-depend on the renderer port.
}
\value{
A list with \code{x}, \code{y}.
}
\keyword{internal}
