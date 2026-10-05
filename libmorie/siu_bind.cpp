// SPDX-License-Identifier: AGPL-3.0-or-later
// nanobind bindings for the SIU report parser vendored from
// rmoriebricklayer (libmorie/siu/, canonical there; edit there first).
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "siu/siu_parse.h"
#include "siu/siu_resolve.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// The parser works on bytes and clips some fields by byte count ("[^\n.]{0,80}"),
// so a value can end inside a multi-byte UTF-8 sequence. nanobind\x27s default
// std::string caster is strict and raised UnicodeDecodeError on such a report;
// decode with replacement, as the pure-Python twin\x27s _s() does.
nb::object lossy_str(const std::string &s) {
    PyObject *o = PyUnicode_DecodeUTF8(s.data(), static_cast<Py_ssize_t>(s.size()), "replace");
    if (!o) throw nb::python_error();
    return nb::steal(o);
}

nb::dict fields_to_dict(const siu::ParsedFields &f) {
    nb::dict d;
    for (const auto &kv : f) d[nb::str(kv.first.c_str())] = lossy_str(kv.second);
    return d;
}

}  // namespace

void register_siu(nb::module_ &m) {
    m.def("siu_html_to_text", [](const std::string &html) { return lossy_str(siu::html_to_text(html)); },
          "html"_a, "Strip SIU report HTML to plain text.");
    m.def("siu_parse_report_text", [](const std::string &text) { return fields_to_dict(siu::parse_report_text(text)); },
          "text"_a, "Parse SIU report plain text into the schema fields.");
    m.def("siu_parse_report_html", [](const std::string &html) { return fields_to_dict(siu::parse_report_html(html)); },
          "html"_a, "Parse SIU report HTML into the schema fields.");
    m.def("siu_to_iso_date", &siu::to_iso_date, "human"_a,
          "English/French long date to ISO 8601 (\"\" when unparseable).");
    m.def("siu_resolve_so", [](const std::string &text) {
        siu::SoResolution r = siu::resolve_subject_officials(text);
        return nb::make_tuple(r.count ? nb::cast(*r.count) : nb::none(), lossy_str(r.reason));
    }, "text"_a, "Resolve the subject-official count: (count or None, reason).");
}
