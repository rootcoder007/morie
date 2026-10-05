// SPDX-License-Identifier: AGPL-3.0-or-later
// nanobind bindings for the SIU report parser vendored from
// rmoriebricklayer (libmorie/siu/, canonical there; edit there first).
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>

#include "siu/siu_parse.h"
#include "siu/siu_resolve.h"

namespace nb = nanobind;
using namespace nb::literals;

void register_siu(nb::module_ &m) {
    m.def("siu_html_to_text", &siu::html_to_text, "html"_a,
          "Strip SIU report HTML to plain text.");
    m.def("siu_parse_report_text", &siu::parse_report_text, "text"_a,
          "Parse SIU report plain text into the schema fields.");
    m.def("siu_parse_report_html", &siu::parse_report_html, "html"_a,
          "Parse SIU report HTML into the schema fields.");
    m.def("siu_to_iso_date", &siu::to_iso_date, "human"_a,
          "English/French long date to ISO 8601 (\"\" when unparseable).");
    m.def("siu_resolve_so", [](const std::string &text) {
        siu::SoResolution r = siu::resolve_subject_officials(text);
        return nb::make_tuple(r.count ? nb::cast(*r.count) : nb::none(), r.reason);
    }, "text"_a, "Resolve the subject-official count: (count or None, reason).");
}
