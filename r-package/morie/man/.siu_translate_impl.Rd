% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_translate_impl}
\alias{.siu_translate_impl}
\title{.siu_translate_impl}
\usage{
.siu_translate_impl(target_lang, source_lang, case_numbers, model, fields, cache_dir, progress)
}
\arguments{
\item{target_lang}{Passed to \code{!=}.}
\item{source_lang}{Optional; may be \code{NULL}. Passed to \code{is.null}.}
\item{case_numbers}{Optional; may be \code{NULL}. A vector; its length is taken and its elements indexed.}
\item{model}{Passed to \code{.siu_llm_call}.}
\item{fields}{Iterated over elementwise, with \code{lapply}.}
\item{cache_dir}{Passed to \code{.siu_load_canonical_overrides}.}
\item{progress}{A flag; the body branches on it.}
}
\description{
Part of the siu implementation; see the file header for the source it follows.
}
\value{
Invisibly,the value of \code{out}, as built in the body.
}
\keyword{internal}
