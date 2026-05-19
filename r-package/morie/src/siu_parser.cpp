// SPDX-License-Identifier: AGPL-3.0-or-later
//
// siu_parser.cpp -- C/C++ parser for the Ontario Special Investigations
// Unit (SIU) corpus: director's reports and news releases.
//
// HTTP(S) transport is libcurl (linked via src/Makevars). This file is
// the foundation; the concurrent fetcher and the 64-field HTML parser
// are layered on top in later commits.

#include <Rcpp.h>
#include <curl/curl.h>
#include <algorithm>
#include <map>
#include <regex>
#include <string>
#include <vector>

// parser_version stamped into every emitted row.
#define MORIE_SIU_PARSER_VERSION "0.2.0"

// # nocov start
// The libcurl HTTP-transport layer below (write_cb, siu_http_get,
// setup_handle, siu_http_get_many) is network code: it is exercised by
// the network-gated tests in test-siu.R, but cannot run under covr /
// R CMD check, which execute offline. Excluded from coverage rather
// than counted as permanently-uncovered.
namespace {

// One-time libcurl global initialisation (libcurl requires this before
// any handle is created when the program is multi-threaded).
struct CurlGlobal {
  CurlGlobal()  { curl_global_init(CURL_GLOBAL_DEFAULT); }
  ~CurlGlobal() { curl_global_cleanup(); }
};
const CurlGlobal kCurlGlobal;

const char* kUserAgent =
  "morie/0.9.5 (+https://github.com/hadesllm/morie)";

// libcurl write callback: append received bytes to a std::string.
size_t write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
  std::string* buf = static_cast<std::string*>(userdata);
  const size_t n = size * nmemb;
  buf->append(ptr, n);
  return n;
}

}  // namespace

//' Fetch a single URL over HTTP(S) via libcurl
//'
//' Internal building block of the SIU parser. Returns the response
//' body, or an empty string on any transport-level failure.
//'
//' @param url URL to fetch.
//' @param timeout_s Request timeout in seconds.
//' @return The response body as a length-1 character vector.
//' @keywords internal
// [[Rcpp::export(.siu_http_get)]]
std::string siu_http_get(std::string url, int timeout_s = 60) {
  CURL* h = curl_easy_init();
  if (h == nullptr) return std::string();
  std::string buf;
  curl_easy_setopt(h, CURLOPT_URL, url.c_str());
  curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, write_cb);
  curl_easy_setopt(h, CURLOPT_WRITEDATA, &buf);
  curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(h, CURLOPT_TIMEOUT, static_cast<long>(timeout_s));
  curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 30L);
  curl_easy_setopt(h, CURLOPT_USERAGENT, kUserAgent);
  curl_easy_setopt(h, CURLOPT_ACCEPT_ENCODING, "");  // all supported
  curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
  const CURLcode rc = curl_easy_perform(h);
  curl_easy_cleanup(h);
  if (rc != CURLE_OK) return std::string();
  return buf;
}

//' libcurl version string morie was built against
//' @return A length-1 character vector.
//' @keywords internal
// [[Rcpp::export(.siu_curl_version)]]
std::string siu_curl_version() {
  return std::string(curl_version());
}

namespace {

// One in-flight request: its index in the input vector and its body buffer.
struct Req {
  int idx;
  std::string body;
};

// Configure a fresh easy handle for one SIU page fetch.
void setup_handle(CURL* e, const char* url, Req* r, long timeout_s) {
  curl_easy_setopt(e, CURLOPT_URL, url);
  curl_easy_setopt(e, CURLOPT_WRITEFUNCTION, write_cb);
  curl_easy_setopt(e, CURLOPT_WRITEDATA, &r->body);
  curl_easy_setopt(e, CURLOPT_PRIVATE, r);
  curl_easy_setopt(e, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(e, CURLOPT_TIMEOUT, timeout_s);
  curl_easy_setopt(e, CURLOPT_CONNECTTIMEOUT, 30L);
  curl_easy_setopt(e, CURLOPT_USERAGENT, kUserAgent);
  curl_easy_setopt(e, CURLOPT_ACCEPT_ENCODING, "");
  curl_easy_setopt(e, CURLOPT_NOSIGNAL, 1L);
}

}  // namespace

//' Fetch many URLs concurrently via the libcurl multi interface
//'
//' Drives up to \code{concurrency} simultaneous transfers; as each
//' finishes the next URL is started, so the connection pool stays
//' saturated. Failed transfers yield an empty string at their slot.
//'
//' @param urls Character vector of URLs.
//' @param concurrency Maximum simultaneous transfers.
//' @param timeout_s Per-request timeout in seconds.
//' @return A character vector of response bodies, parallel to \code{urls}.
//' @keywords internal
// [[Rcpp::export(.siu_http_get_many)]]
Rcpp::CharacterVector siu_http_get_many(Rcpp::CharacterVector urls,
                                        int concurrency = 16,
                                        int timeout_s = 60) {
  const int n = urls.size();
  Rcpp::CharacterVector out(n);
  for (int i = 0; i < n; ++i) out[i] = "";
  if (n == 0) return out;
  if (concurrency < 1) concurrency = 1;
  if (concurrency > n) concurrency = n;

  CURLM* multi = curl_multi_init();
  std::vector<Req*> reqs;
  reqs.reserve(n);
  const long tmo = static_cast<long>(timeout_s);
  int next = 0;
  int in_flight = 0;

  while (next < n && in_flight < concurrency) {
    Req* r = new Req{next, std::string()};
    reqs.push_back(r);
    CURL* e = curl_easy_init();
    setup_handle(e, std::string(urls[next]).c_str(), r, tmo);
    curl_multi_add_handle(multi, e);
    ++next;
    ++in_flight;
  }

  do {
    int still_running = 0;
    curl_multi_perform(multi, &still_running);
    int numfds = 0;
    curl_multi_poll(multi, nullptr, 0, 1000, &numfds);

    CURLMsg* msg = nullptr;
    int msgs_left = 0;
    while ((msg = curl_multi_info_read(multi, &msgs_left)) != nullptr) {
      if (msg->msg != CURLMSG_DONE) continue;
      CURL* e = msg->easy_handle;
      Req* r = nullptr;
      curl_easy_getinfo(e, CURLINFO_PRIVATE, &r);
      if (r != nullptr && msg->data.result == CURLE_OK) {
        out[r->idx] = r->body;
      }
      curl_multi_remove_handle(multi, e);
      curl_easy_cleanup(e);
      --in_flight;
      if (next < n) {
        Req* nr = new Req{next, std::string()};
        reqs.push_back(nr);
        CURL* ne = curl_easy_init();
        setup_handle(ne, std::string(urls[next]).c_str(), nr, tmo);
        curl_multi_add_handle(multi, ne);
        ++next;
        ++in_flight;
      }
    }
    Rcpp::checkUserInterrupt();
  } while (in_flight > 0 || next < n);

  curl_multi_cleanup(multi);
  for (Req* r : reqs) delete r;
  return out;
}
// # nocov end

// ===========================================================================
// HTML parsing -- SIU director's-report pages -> the 64-column schema.
// ===========================================================================

namespace {

// Canonical 64-column order of the SIU dataset.
const std::vector<std::string> kSiuCols = {
  "case_number", "drid", "nrid", "source_url_report", "source_url_news",
  "scraped_at_utc", "parser_version", "date_of_incident_iso",
  "date_of_incident_raw", "time_of_incident_raw", "date_of_injury_iso",
  "date_of_injury_raw", "incident_to_injury_raw", "date_siu_notified_iso",
  "date_siu_notified_raw", "time_of_notification_raw", "notifying_party",
  "notifying_party_other_text", "date_of_director_decision_iso",
  "date_of_director_decision_raw", "time_of_director_decision_raw",
  "siu_investigators", "siu_forensics_investigators", "police_service",
  "number_of_officers_involved", "location_of_call",
  "type_of_building_or_scene", "reason_for_interaction", "injuries_sustained",
  "injuries_other_text", "specific_injuries", "location_of_treatment",
  "number_of_affected_persons", "sex_gender_affected", "age_affected",
  "affected_interviewed", "date_of_affected_interview_iso",
  "date_of_affected_interview_raw", "number_of_civilian_witnesses",
  "date_of_witness_interview_raw", "number_of_subject_officials",
  "subject_official_interviewed_or_notes", "date_of_subject_interview_raw",
  "number_of_witness_officials", "date_of_witness_official_interview_raw",
  "evidence_types", "evidence_other_text", "evidence_features",
  "narrative_summary", "relevant_legislation", "legislation_other_text",
  "weapons_or_force_used", "weapons_other_text", "charges_recommended",
  "directors_decision_reasonable", "supplemental_materials",
  "news_links_extra", "mental_health_or_race_indications", "_language",
  "news_release_title", "news_release_date_iso", "news_release_date_raw",
  "news_release_summary", "directors_name"
};

// Decode the HTML entities that occur in SIU pages.
std::string decode_entities(std::string s) {
  struct E { const char* k; const char* v; };
  static const E named[] = {
    {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""},
    {"&apos;", "'"}, {"&nbsp;", " "}, {"&rsquo;", "'"}, {"&lsquo;", "'"},
    {"&ldquo;", "\""}, {"&rdquo;", "\""}, {"&ndash;", "-"}, {"&mdash;", "-"},
    {"&hellip;", "..."}, {"&eacute;", "\xC3\xA9"}, {"&egrave;", "\xC3\xA8"},
    {"&agrave;", "\xC3\xA0"}, {"&ccedil;", "\xC3\xA7"}, {"&acirc;", "\xC3\xA2"},
    {"&ecirc;", "\xC3\xAA"}, {"&icirc;", "\xC3\xAE"}, {"&ocirc;", "\xC3\xB4"},
    {"&ucirc;", "\xC3\xBB"}, {"&iuml;", "\xC3\xAF"}, {"&euml;", "\xC3\xAB"},
    {"&ugrave;", "\xC3\xB9"}, {"&igrave;", "\xC3\xAC"}, {"&Eacute;", "\xC3\x89"},
    {"&oelig;", "\xC5\x93"}, {"&laquo;", "\xC2\xAB"}, {"&raquo;", "\xC2\xBB"},
    {"&#39;", "'"}
  };
  for (const E& e : named) {
    std::string::size_type p = 0;
    while ((p = s.find(e.k, p)) != std::string::npos) {
      s.replace(p, std::string(e.k).size(), e.v);
      p += std::string(e.v).size();
    }
  }
  // Numeric entities &#NN; / &#xNN; -- decode the ASCII range; map a few
  // common Unicode punctuation points to their ASCII equivalents.
  static const std::regex num_re("&#(x?[0-9A-Fa-f]+);");
  std::string out;
  out.reserve(s.size());
  auto begin = std::sregex_iterator(s.begin(), s.end(), num_re);
  auto end = std::sregex_iterator();
  std::string::size_type last = 0;
  for (auto it = begin; it != end; ++it) {
    out.append(s, last, it->position() - last);
    std::string tok = (*it)[1].str();
    long cp = (tok[0] == 'x') ? std::strtol(tok.c_str() + 1, nullptr, 16)
                              : std::strtol(tok.c_str(), nullptr, 10);
    if (cp == 8217 || cp == 8216 || cp == 8242) out.push_back('\'');
    else if (cp == 8220 || cp == 8221) out.push_back('"');
    else if (cp == 8211 || cp == 8212) out.push_back('-');
    else if (cp >= 32 && cp < 127) out.push_back(static_cast<char>(cp));
    else out.push_back(' ');
    last = it->position() + it->length();
  }
  out.append(s, last, std::string::npos);
  return out;
}

// Collapse runs of whitespace to single spaces and trim.
std::string squeeze(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  bool sp = false;
  for (char c : s) {
    const bool ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                     c == '\f' || c == '\v');
    if (ws) { sp = true; continue; }
    if (sp && !out.empty()) out.push_back(' ');
    sp = false;
    out.push_back(c);
  }
  return out;
}

// Strip all HTML markup from a fragment and return decoded plain text.
std::string html_to_text(std::string h) {
  h = std::regex_replace(h, std::regex("<script[^>]*>.*?</script>",
                                       std::regex::icase), " ");
  h = std::regex_replace(h, std::regex("<style[^>]*>.*?</style>",
                                       std::regex::icase), " ");
  h = std::regex_replace(h, std::regex("<[^>]+>"), " ");
  return squeeze(decode_entities(h));
}

// Plain text of the report section whose <h2> carries id="section_<n>".
std::string section_text(const std::string& html, int n) {
  const std::string anchor = "id=\"section_" + std::to_string(n) + "\"";
  std::string::size_type a = html.find(anchor);
  if (a == std::string::npos) return std::string();
  std::string::size_type body = html.find('>', a);
  if (body == std::string::npos) return std::string();
  ++body;
  std::string::size_type b = html.find("<h2", body);
  if (b == std::string::npos) b = html.size();
  return html_to_text(html.substr(body, b - body));
}

// First capture group of `pat` in `text`, or "" when there is no match.
std::string rx1(const std::string& text, const std::string& pat) {
  try {
    std::smatch m;
    const std::regex re(pat, std::regex::icase);
    if (std::regex_search(text, m, re) && m.size() > 1) return m[1].str();
  } catch (...) {}
  return std::string();
}

// Highest N among "<label> #N" tokens; 1 if the bare label occurs; else "".
std::string label_count(const std::string& text, const std::string& label) {
  int hi = 0;
  try {
    const std::regex re(label + "\\s*#\\s*(\\d+)");
    auto begin = std::sregex_iterator(text.begin(), text.end(), re);
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
      const int v = std::atoi((*it)[1].str().c_str());
      if (v > hi) hi = v;
    }
  } catch (...) {}
  if (hi > 0) return std::to_string(hi);
  try {
    if (std::regex_search(text, std::regex("\\b" + label + "\\b")))
      return "1";
  } catch (...) {}
  return std::string();
}

const char* kMonths[] = {"january", "february", "march", "april", "may",
                         "june", "july", "august", "september", "october",
                         "november", "december"};

// "May 8, 2026" or "8 May, 2026" -> "2026-05-08"; "" when unparseable.
std::string to_iso(const std::string& raw) {
  std::smatch m;
  std::string mon, day, year;
  if (std::regex_search(raw, m,
        std::regex("([A-Za-z]+)\\s+(\\d{1,2}),?\\s+(\\d{4})"))) {
    mon = m[1].str(); day = m[2].str(); year = m[3].str();
  } else if (std::regex_search(raw, m,
        std::regex("(\\d{1,2})\\s+([A-Za-z]+),?\\s+(\\d{4})"))) {
    day = m[1].str(); mon = m[2].str(); year = m[3].str();
  } else {
    return std::string();
  }
  std::transform(mon.begin(), mon.end(), mon.begin(), ::tolower);
  int mi = 0;
  for (int i = 0; i < 12; ++i) if (mon == kMonths[i]) { mi = i + 1; break; }
  if (mi == 0) return std::string();
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%s-%02d-%02d", year.c_str(), mi,
                std::atoi(day.c_str()));
  return std::string(buf);
}

// First "Month D, YYYY" date appearing in `text`.
std::string first_date(const std::string& text) {
  return rx1(text, "([A-Za-z]+\\s+\\d{1,2},?\\s+\\d{4})");
}

// Every "Month D, YYYY" date in `text`, in order of appearance.
std::vector<std::string> all_dates(const std::string& text) {
  std::vector<std::string> v;
  try {
    const std::regex re("[A-Za-z]+\\s+\\d{1,2},?\\s+\\d{4}");
    for (auto it = std::sregex_iterator(text.begin(), text.end(), re);
         it != std::sregex_iterator(); ++it)
      v.push_back(it->str());
  } catch (...) {}
  return v;
}

// Modal police-service name in `text`: the most-mentioned "X Police"
// / "X Police Service" (ties broken toward the longer, more complete
// name), with SIU self-references and a leading "The " dropped. Far
// more robust than the first regex hit, which is often truncated.
std::string modal_service(const std::string& text) {
  std::map<std::string, int> counts;
  try {
    const std::regex re("[A-Z][A-Za-z.'\\-]+(?: [A-Z][A-Za-z.'\\-]+){0,4} "
                        "Police(?: Service)?");
    for (auto it = std::sregex_iterator(text.begin(), text.end(), re);
         it != std::sregex_iterator(); ++it) {
      std::string s = it->str();
      if (s.find("SIU") != std::string::npos) continue;
      if (s.rfind("The ", 0) == 0) s = s.substr(4);
      ++counts[s];
    }
  } catch (...) {}
  std::string best;
  int best_n = 0;
  for (const auto& kv : counts) {
    if (kv.second > best_n ||
        (kv.second == best_n && kv.first.size() > best.size())) {
      best_n = kv.second;
      best = kv.first;
    }
  }
  return best;
}

}  // namespace

//' Parse one SIU director's-report HTML page into the 64-column schema
//'
//' @param html The report page HTML.
//' @param drid The director's-report id.
//' @param url The source URL of the report page.
//' @return A named character vector with the 64 SIU dataset columns;
//'   report-derived fields are populated, news fields left empty.
//' @keywords internal
// [[Rcpp::export(.siu_parse_report)]]
Rcpp::CharacterVector siu_parse_report(std::string html, int drid,
                                       std::string url) {
  std::map<std::string, std::string> f;
  f["drid"] = std::to_string(drid);
  f["source_url_report"] = url;
  f["parser_version"] = MORIE_SIU_PARSER_VERSION;
  {
    std::time_t t = std::time(nullptr);
    std::tm gm;
#ifdef _WIN32
    gmtime_s(&gm, &t);
#else
    gmtime_r(&t, &gm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &gm);
    f["scraped_at_utc"] = buf;
  }

  // Language: English vs French boilerplate header.
  if (html.find("Mandate of the SIU") != std::string::npos)
    f["_language"] = "en";
  else if (html.find("Mandat de l'UES") != std::string::npos ||
           html.find("Mandat de l\xE2\x80\x99UES") != std::string::npos)
    f["_language"] = "fr";
  else
    f["_language"] = "unknown";

  const std::string full = html_to_text(html);
  f["case_number"] = rx1(full, "Case #\\s*([0-9A-Za-z]+-[0-9A-Za-z]+-[0-9]+)");

  const std::string investigation = section_text(html, 4);
  const std::string narrative = section_text(html, 6);
  const std::string legislation = section_text(html, 7);
  std::string analysis = section_text(html, 8);
  if (analysis.empty()) analysis = section_text(html, 7);  // older layout

  f["narrative_summary"] = narrative;
  if (!legislation.empty()) f["relevant_legislation"] = legislation;

  // SIU notification: first date in "The Investigation"; the notifying
  // service is the police service named before "contacted the SIU".
  const std::string notif_date = first_date(investigation);
  if (!notif_date.empty()) {
    f["date_siu_notified_raw"] = notif_date;
    f["date_siu_notified_iso"] = to_iso(notif_date);
  }
  std::string service = modal_service(investigation);
  if (service.empty()) service = modal_service(full);
  if (!service.empty()) {
    f["police_service"] = service;
    f["notifying_party"] = service;
  }

  // Incident date: in "The Investigation" the first date is the SIU
  // notification and the second is the incident itself; fall back to
  // the Incident Narrative, then the Analysis section.
  std::string inc_date;
  {
    const std::vector<std::string> idates = all_dates(investigation);
    if (idates.size() >= 2) inc_date = idates[1];
    if (inc_date.empty()) inc_date = first_date(narrative);
    if (inc_date.empty()) inc_date = first_date(analysis);
    if (inc_date.empty() && !idates.empty()) inc_date = idates[0];
  }
  if (!inc_date.empty()) {
    f["date_of_incident_raw"] = inc_date;
    f["date_of_incident_iso"] = to_iso(inc_date);
  }

  // Director's decision date + name from the signature block.
  const std::string dec_date = rx1(
    full, "Date:\\s*([A-Za-z]+\\s+\\d{1,2},?\\s+\\d{4})");
  if (!dec_date.empty()) {
    f["date_of_director_decision_raw"] = dec_date;
    f["date_of_director_decision_iso"] = to_iso(dec_date);
  }
  std::string dname = rx1(
    full, "(?:approved by|signed by)\\s+([A-Z][A-Za-z.'\\- ]+?)\\s+Director");
  if (dname.empty())
    dname = rx1(full, "([A-Z][a-z]+(?: [A-Z]\\.?)? [A-Z][a-z]+),?\\s+"
                      "Director\\s+Special Investigations Unit");
  if (dname.empty())
    dname = rx1(full, "Director,?\\s+Special Investigations Unit\\s+"
                      "([A-Z][a-z]+(?: [A-Z]\\.?)? [A-Z][a-z]+)");
  f["directors_name"] = dname;

  // Official / witness counts.
  const std::string so = label_count(investigation + " " + narrative, "SO");
  const std::string wo = label_count(investigation + " " + narrative, "WO");
  const std::string cw = label_count(investigation + " " + narrative, "CW");
  f["number_of_subject_officials"] = so;
  f["number_of_witness_officials"] = wo;
  f["number_of_civilian_witnesses"] = cw;
  if (!so.empty()) {
    int tot = std::atoi(so.c_str());
    if (!wo.empty()) tot += std::atoi(wo.c_str());
    f["number_of_officers_involved"] = std::to_string(tot);
  }

  // Affected-person attributes from the narrative.
  const std::string age = rx1(full, "(\\d{1,3})[ -]year[ -]old");
  if (!age.empty()) f["age_affected"] = age;
  {
    const std::string n = narrative + " " + analysis;
    auto count = [&](const std::string& w) {
      int c = 0;
      try {
        const std::regex re("\\b" + w + "\\b", std::regex::icase);
        for (auto it = std::sregex_iterator(n.begin(), n.end(), re);
             it != std::sregex_iterator(); ++it) ++c;
      } catch (...) {}
      return c;
    };
    const int male = count("he") + count("his") + count("him") +
                     count("man") + count("boy") + count("male");
    const int female = count("she") + count("her") + count("woman") +
                       count("girl") + count("female");
    const std::string lc = full;  // a non-binary signal anywhere counts
    const bool nonbinary =
      lc.find("non-binary") != std::string::npos ||
      lc.find("nonbinary") != std::string::npos ||
      lc.find("transgender") != std::string::npos ||
      lc.find("two-spirit") != std::string::npos;
    // Sex/gender is not binary: an explicit non-binary / transgender /
    // two-spirit signal wins; otherwise the dominant gendered-term
    // count decides; an indeterminate page is left blank.
    if (nonbinary)
      f["sex_gender_affected"] = "Non-binary";
    else if (male + female >= 3)
      f["sex_gender_affected"] = (male >= female) ? "Male" : "Female";
  }
  f["location_of_call"] = squeeze(rx1(full,
    "in the area of (.{5,110}?,\\s*[A-Z][a-z]+)"));

  // Director's decision outcome.
  if (analysis.find("no reasonable grounds") != std::string::npos)
    f["directors_decision_reasonable"] = "No";
  else if (analysis.find("reasonable grounds to believe") != std::string::npos)
    f["directors_decision_reasonable"] = "Yes";
  {
    const std::string ch = rx1(analysis,
      "(no charges?[^.]{0,60}|charges? (?:will|would|have|has|were|was)"
      "[^.]{0,60}|the charge of [^.]{0,60})");
    if (!ch.empty()) f["charges_recommended"] = squeeze(ch);
  }

  // Mental-health / race indications mentioned in the narrative.
  {
    const std::string n = full;
    std::string tags;
    const char* kw[] = {"mental health", "Black", "Indigenous",
                        "First Nation", "racializ", "racial", "in crisis"};
    for (const char* k : kw) {
      if (n.find(k) != std::string::npos) {
        if (!tags.empty()) tags += "; ";
        tags += k;
      }
    }
    f["mental_health_or_race_indications"] = tags;
  }

  // News release linked from the report page: capture both the title
  // and the nrid, so the news page can be fetched and joined later.
  {
    std::smatch m;
    if (std::regex_search(html, m, std::regex(
          "News Releases for this Case:.*?<a[^>]*href=\"([^\"]+)\"[^>]*>"
          "(.*?)</a>", std::regex::icase))) {
      f["news_release_title"] = squeeze(html_to_text(m[2].str()));
      const std::string nrid = rx1(m[1].str(), "nrid=(\\d+)");
      if (!nrid.empty()) {
        f["nrid"] = nrid;
        f["source_url_news"] =
          "https://www.siu.on.ca/en/news_template.php?nrid=" + nrid;
      }
    }
  }

  Rcpp::CharacterVector out(kSiuCols.size());
  Rcpp::CharacterVector nm(kSiuCols.size());
  for (std::size_t i = 0; i < kSiuCols.size(); ++i) {
    nm[i] = kSiuCols[i];
    const auto it = f.find(kSiuCols[i]);
    out[i] = (it == f.end()) ? "" : it->second;
  }
  out.attr("names") = nm;
  return out;
}

//' Parse one SIU news-release HTML page
//'
//' @param html The news-release page HTML.
//' @param nrid The news-release id.
//' @param url The source URL of the news-release page.
//' @return A named character vector: nrid, source_url_news,
//'   news_release_title, news_release_date_iso, news_release_date_raw,
//'   news_release_summary.
//' @keywords internal
// [[Rcpp::export(.siu_parse_news)]]
Rcpp::CharacterVector siu_parse_news(std::string html, int nrid,
                                     std::string url) {
  std::string title;
  try {
    const std::regex hre("<h[1-4][^>]*>(.*?)</h[1-4]>",
                         std::regex::icase);
    for (auto it = std::sregex_iterator(html.begin(), html.end(), hre);
         it != std::sregex_iterator(); ++it) {
      const std::string t = squeeze(html_to_text((*it)[1].str()));
      if (t.size() > title.size() &&
          t.find("News Release") == std::string::npos &&
          t != "The Unit")
        title = t;
    }
  } catch (...) {}

  // Dateline: "<strong>City, ON</strong> ( DD Month, YYYY ) ---".
  std::string draw = rx1(html, "</strong>\\s*\\(([^)]{6,32})\\)\\s*-");
  if (draw.empty())
    draw = rx1(html_to_text(html),
               "\\(([0-9]{1,2} [A-Za-z]+,? [0-9]{4})\\)");

  const std::string txt = html_to_text(html);
  std::string summary = rx1(
    txt, "\\)\\s*-+\\s*(.{20,700}?)(?:Full Director|If you or someone)");
  if (summary.empty())
    summary = rx1(txt, "\\)\\s*-+\\s*(.{20,400})");

  std::map<std::string, std::string> f;
  f["nrid"] = std::to_string(nrid);
  f["source_url_news"] = url;
  f["news_release_title"] = title;
  f["news_release_date_raw"] = squeeze(draw);
  f["news_release_date_iso"] = to_iso(draw);
  f["news_release_summary"] = squeeze(summary);

  static const char* cols[] = {
    "nrid", "source_url_news", "news_release_title",
    "news_release_date_iso", "news_release_date_raw",
    "news_release_summary"};
  Rcpp::CharacterVector out(6);
  Rcpp::CharacterVector nm(6);
  for (int i = 0; i < 6; ++i) { nm[i] = cols[i]; out[i] = f[cols[i]]; }
  out.attr("names") = nm;
  return out;
}
