// SPDX-License-Identifier: AGPL-3.0-or-later
// nanobind bindings for the SIU report parser vendored from
// rmoriebricklayer (libmorie/siu/, canonical there; edit there first).
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <stdexcept>
#include <string>

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

// The caps rmoriebricklayer's own entry points apply (rmbl_siu.cpp): a report
// page is a few hundred KB, and the core's regexes run over text whose lines
// normalize_text() has capped (libstdc++'s regex executor recurses once per
// character a repeated atom consumes). nanobind turns the exception into a
// ValueError.
const std::string &checked_page(const std::string &s, const char *what) {
    if (s.size() > (2u << 20)) throw std::invalid_argument(std::string(what) + " is larger than 2 MiB: not a report page");
    return s;
}

}  // namespace

void register_siu(nb::module_ &m) {
    m.def("siu_html_to_text", [](const std::string &html) { return lossy_str(siu::html_to_text(checked_page(html, "html"))); },
          "html"_a, "Strip SIU report HTML to plain text.");
    m.def("siu_parse_report_text", [](const std::string &text) {
        return fields_to_dict(siu::parse_report_text(siu::normalize_text(checked_page(text, "text"))));
    }, "text"_a, "Parse SIU report plain text into the schema fields.");
    m.def("siu_parse_report_html", [](const std::string &html) { return fields_to_dict(siu::parse_report_html(checked_page(html, "html"))); },
          "html"_a, "Parse SIU report HTML into the schema fields.");
    m.def("siu_to_iso_date", [](const std::string &human) { return human.size() > 4096 ? std::string() : siu::to_iso_date(human); },
          "human"_a, "English/French long date to ISO 8601 (\"\" when unparseable).");
    m.def("siu_resolve_so", [](const std::string &text) {
        siu::SoResolution r = siu::resolve_subject_officials(siu::normalize_text(checked_page(text, "text")));
        return nb::make_tuple(r.count ? nb::cast(*r.count) : nb::none(), lossy_str(r.reason));
    }, "text"_a, "Resolve the subject-official count: (count or None, reason).");
}
