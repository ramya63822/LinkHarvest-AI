# LinkHarvest AI — Profile Dossier Generator

Paste one or more profile URLs and get an organised PDF dossier for each. LinkedIn profiles are read through Apify; every other URL is scraped directly and interpreted by Gemini. One URL returns a PDF, several return a zip.

## Architecture

| Layer | Technology |
|---|---|
| Web framework | FastAPI (ASGI, served by Uvicorn) |
| Page fetching | httpx (async, concurrent) |
| HTML → text | BeautifulSoup + lxml |
| LinkedIn profiles | Apify actor (structured JSON, no Gemini call) |
| Other pages | Google Gemini with a JSON response schema |
| PDF generation | ReportLab |

## How it works

1. **Fetch.** Each URL is downloaded concurrently with httpx, following redirects.
2. **Reduce to text.** BeautifulSoup strips `script`, `style`, `nav`, `header`, `footer` and other chrome, leaving readable text (capped at 60,000 characters).
3. **Extract.** Gemini fills a fixed JSON schema — name, headline, company, location, summary, experience, education, skills. Using a response schema rather than parsing a text blob with regexes is what lets the PDF lay out real sections.
4. **Render.** ReportLab writes an A4 dossier: header block, fact table, summary, experience entries with dates and descriptions, education, and skills.
5. **Deliver.** One result downloads as a PDF; several are zipped.

Failures are reported per URL, so one bad link never sinks the batch.

## What works

| URL type | How it is read |
|---|---|
| **LinkedIn profiles** | Apify runs a LinkedIn scraper on its own infrastructure and returns structured JSON |
| **Everything else** | Fetched directly, reduced to text, and interpreted by Gemini |

LinkedIn cannot be scraped directly: it answers non-browser requests with HTTP 999 and an
empty body, so there is no text to parse. Routing through Apify solves this without your
own LinkedIn account or cookies being involved anywhere — which matters, because
cookie-based automation is what leaked credentials in this project's history.

All LinkedIn URLs in a batch go out as **one** Apify run, which is faster and cheaper than
one call each. Without `APIFY` set, LinkedIn URLs are refused with an explanation and
everything else still works.

Pages rendered entirely by JavaScript yield little text; the app reports this rather than
guessing. Facebook, Instagram and X are refused, since they require a login.

### Apify setup

1. Sign up at [apify.com](https://apify.com) — the free plan gives $5/month of credit and
   needs no card.
2. Copy your token from **Settings → Integrations**.
3. Put it in `.env` as `APIFY=...`.

The default actor is [`harvestapi~linkedin-profile-scraper`](https://apify.com/harvestapi/linkedin-profile-scraper)
at roughly **$0.004 per profile**, so $5 covers about 1,200 profiles. Override it with
`APIFY_ACTOR` and `APIFY_MODE` if you prefer another.

Pick actors carefully. Some demand **full access to your Apify account** before they will
run — the default here does not.

### A note on legality

Scraping LinkedIn is contrary to its [User Agreement](https://www.linkedin.com/legal/user-agreement)
however it is done. Using Apify moves the scraping onto a third party's infrastructure
rather than your LinkedIn account, which is a better position but not a clean one. LinkedIn
sued Proxycurl, then the largest provider in this space, and it shut down in July 2025.

### A note on the User-Agent

For non-LinkedIn pages the scraper identifies itself honestly rather than impersonating
Chrome. This is functional, not just polite: Wikipedia returns **403** with a link to its
robot policy when a script claims to be a browser, and **200** when it says what it is.
`robots.txt` should be respected for any serious use.

## Requirements

- Python 3.9+
- A Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- An Apify token from [apify.com](https://apify.com), only if you need LinkedIn

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

# Create .env next to app.py:
#   GEMINI_API_KEY=your_key_here
#   APIFY=your_apify_token        (optional, for LinkedIn)

python app.py
```

Open <http://127.0.0.1:8000>.

The app refuses to start without `GEMINI_API_KEY`, and says so.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Gemini API key |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Override if the default model is retired |
| `APIFY` | *(optional)* | Apify token; without it LinkedIn URLs are refused |
| `APIFY_ACTOR` | `harvestapi~linkedin-profile-scraper` | Which Apify actor to run |
| `APIFY_MODE` | `Profile details no email ($4 per 1k)` | The actor's scrape mode |
| `APIFY_TIMEOUT` | `300` | Apify sync-run timeout; its hard limit is 300s |
| `MAX_URLS` | `10` | URLs accepted per batch |
| `MAX_CONCURRENCY` | `2` | Parallel page+model workers |
| `FETCH_TIMEOUT` | `25` | Per-page timeout, seconds |
| `USER_AGENT` | *(identifying string)* | Sent with every request |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address and port |

Gemini's free tier is rate limited. If batches fail with `429 quota exceeded`, lower `MAX_CONCURRENCY` or space out requests.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | The dashboard |
| `POST` | `/extract` | Takes a newline-separated `urls` field; returns per-URL JSON results and a download link |
| `GET` | `/download/{job_id}` | Returns the PDF or zip for a completed job |

## Limitations

- Download jobs are held in memory, so they are lost on restart and will not work across multiple workers.
- Generated files accumulate in `generated/`; nothing prunes them.
- There is no authentication — do not expose a deployed instance publicly without adding some.
- Extraction quality depends on how much real text the page serves.
- Apify's synchronous endpoint returns HTTP 408 past 300 seconds, which caps how many LinkedIn profiles one batch can carry.
- Scraper actors break when LinkedIn changes its markup; failures are reported per URL.