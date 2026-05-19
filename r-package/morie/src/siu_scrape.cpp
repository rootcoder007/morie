// SPDX-License-Identifier: AGPL-3.0-or-later
//
// siu_scrape.cpp -- C/C++ scraper for the Ontario Special Investigations
// Unit (SIU) corpus: director's reports and news releases.
//
// HTTP(S) transport is libcurl (linked via src/Makevars). This file is
// the foundation; the concurrent fetcher and the 64-field HTML parser
// are layered on top in later commits.

#include <Rcpp.h>
#include <curl/curl.h>
#include <string>
#include <vector>

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
//' Internal building block of the SIU scraper. Returns the response
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
