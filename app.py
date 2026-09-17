import asyncio
import io
import os
import re
import uuid
import zipfile
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from google.genai import errors as genai_errors

load_dotenv()

app = FastAPI(title="LinkHarvest AI — Profile Dossier Generator")
templates = Jinja2Templates(directory="templates")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Create a .env file next to app.py containing:\n"
        "    GEMINI_API_KEY=your_key_here\n"
        "Get a key at https://aistudio.google.com/apikey"
    )

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

GENERATED_DIR = "generated"
os.makedirs(GENERATED_DIR, exist_ok=True)

MAX_URLS = int(os.getenv("MAX_URLS", "10"))
FETCH_TIMEOUT = float(os.getenv("FETCH_TIMEOUT", "25"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))

RETRYABLE_CODES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3

# Apify runs LinkedIn scrapers on its own infrastructure and returns structured
# JSON, so LinkedIn profiles skip Gemini entirely — cheaper, faster and more
# accurate than asking a model to infer structure from scraped text.
# Without a token, LinkedIn URLs are refused with an explanation instead.
APIFY_TOKEN = os.getenv("APIFY")
APIFY_ACTOR = os.getenv("APIFY_ACTOR", "harvestapi~linkedin-profile-scraper")
APIFY_MODE = os.getenv("APIFY_MODE", "Profile details no email ($4 per 1k)")
APIFY_TIMEOUT = float(os.getenv("APIFY_TIMEOUT", "300"))

# An honest, identifying User-Agent. This is not just politeness: Wikipedia
# returns 403 with a link to its robot policy when a script pretends to be
# Chrome, and 200 when it says what it is. Sites that want to refuse bots
# can then do so deliberately.
USER_AGENT = os.getenv(
    "USER_AGENT",
    "LinkHarvestAI/6.0 (profile dossier generator; "
    "+https://github.com/Prajin22/LinkHarvest-AI)",
)

HTTP_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

LINKEDIN_HOST = "linkedin.com"

NO_APIFY_MESSAGE = (
    "LinkedIn blocks server-side scraping: it answers non-browser requests "
    "with HTTP 999 and an empty body, so there is no profile text to read. "
    "Set APIFY in .env to route LinkedIn URLs through an Apify scraper."
)

# Sites that serve an anti-bot wall instead of content to plain HTTP clients.
# Detected up front so the user gets an explanation rather than an empty dossier.
BLOCKED_HOSTS = {
    "facebook.com": "Facebook requires a login for profile pages.",
    "instagram.com": "Instagram requires a login for profile pages.",
    "x.com": "X/Twitter requires a login to read profiles.",
    "twitter.com": "X/Twitter requires a login to read profiles.",
}

# Structured output beats the old "labelled text block + regex" approach: the
# model fills a schema, so the PDF can lay out real sections instead of a blob.
PROFILE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "full_name": {"type": "STRING"},
        "headline": {"type": "STRING"},
        "company": {"type": "STRING"},
        "location": {"type": "STRING"},
        "email": {"type": "STRING"},
        "summary": {"type": "STRING"},
        "experience": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "organisation": {"type": "STRING"},
                    "dates": {"type": "STRING"},
                    "location": {"type": "STRING"},
                    "skills": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["title", "organisation"],
            },
        },
        "education": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "institution": {"type": "STRING"},
                    "qualification": {"type": "STRING"},
                    "years": {"type": "STRING"},
                },
                "required": ["institution"],
            },
        },
        "skills": {"type": "ARRAY", "items": {"type": "STRING"}},
        "certifications": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "issuer": {"type": "STRING"},
                    "date": {"type": "STRING"},
                },
                "required": ["name"],
            },
        },
        "languages": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "proficiency": {"type": "STRING"},
                },
                "required": ["name"],
            },
        },
        "honors": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "issuer": {"type": "STRING"},
                    "date": {"type": "STRING"},
                },
                "required": ["title"],
            },
        },
        "volunteering": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "role": {"type": "STRING"},
                    "organisation": {"type": "STRING"},
                    "cause": {"type": "STRING"},
                    "dates": {"type": "STRING"},
                },
                "required": ["role"],
            },
        },
    },
    "required": ["full_name", "headline", "experience", "education", "skills"],
}

# job_id -> {"path": str, "filename": str}. In-memory is fine for a single
# process; a multi-worker deployment would need shared storage.
JOBS: dict[str, dict] = {}


def normalise_url(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    if not re.match(r"^https?://", raw, re.IGNORECASE):
        raw = "https://" + raw
    return raw


def host_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def is_linkedin(url: str) -> bool:
    host = host_of(url)
    return host == LINKEDIN_HOST or host.endswith("." + LINKEDIN_HOST)


def linkedin_slug(url: str) -> str:
    """The '/in/<slug>' part, used to match results back to requested URLs."""
    match = re.search(r"/in/([^/?#]+)", urlparse(url).path or "", re.IGNORECASE)
    return match.group(1).lower().rstrip("/") if match else url.lower()


class FetchError(Exception):
    """A page could not be turned into readable text."""


def _text(value) -> str:
    return str(value).strip() if value not in (None, "") else ""


def _format_count(val) -> str:
    if val is None or val == "":
        return ""
    try:
        n = int(val)
        return f"{n:,}"
    except (ValueError, TypeError):
        return str(val).strip()


def map_apify_profile(row: dict) -> dict:
    """Convert one Apify row into a rich structured profile dictionary."""
    name = " ".join(filter(None, [_text(row.get("firstName")), _text(row.get("lastName"))]))
    if not name:
        name = _text(row.get("name")) or _text(row.get("publicIdentifier")) or "Unknown"

    location = row.get("location") or {}
    if isinstance(location, dict):
        location_text = _text(location.get("linkedinText")) or _text(
            (location.get("parsed") or {}).get("text")) or _text(
            ", ".join(filter(None, [
                _text((location.get("parsed") or {}).get("city")),
                _text((location.get("parsed") or {}).get("state")),
                _text((location.get("parsed") or {}).get("country"))
            ]))
        )
    else:
        location_text = _text(location)

    current = (row.get("currentPosition") or [{}])[0] if row.get("currentPosition") else {}
    company = _text(current.get("companyName"))

    # Email extraction (from email search or public contacts)
    emails = row.get("emails") or []
    email = ""
    if isinstance(emails, list) and emails:
        email = _text(emails[0])
    elif isinstance(emails, str):
        email = _text(emails)
    if not email and row.get("email"):
        email = _text(row.get("email"))

    connections = _format_count(row.get("connectionsCount") or row.get("connections"))
    followers = _format_count(row.get("followerCount") or row.get("followers"))
    open_to_work = bool(row.get("openToWork"))
    verified = bool(row.get("verified"))

    experience = []
    for item in (row.get("experience") or []):
        if not isinstance(item, dict):
            continue
        start = _text((item.get("startDate") or {}).get("text") if isinstance(item.get("startDate"), dict) else item.get("startDate"))
        end = _text((item.get("endDate") or {}).get("text") if isinstance(item.get("endDate"), dict) else item.get("endDate"))
        dates = " - ".join(filter(None, [start, end])) or _text(item.get("duration"))
        if start and end and _text(item.get("duration")):
            dates = f"{dates} ({_text(item['duration'])})"

        # Extract role-specific skills as words if present
        role_skills = []
        for s in (item.get("skills") or []):
            if isinstance(s, dict):
                role_skills.append(_text(s.get("name") or s.get("title")))
            elif isinstance(s, str):
                role_skills.append(_text(s))
        role_skills = [s for s in role_skills if s]

        experience.append({
            "title": _text(item.get("position")),
            "organisation": _text(item.get("companyName")),
            "dates": dates,
            "location": _text(item.get("location")),
            "employment_type": _text(item.get("employmentType")),
            "skills": role_skills,
        })

    education = []
    for item in (row.get("education") or []):
        if not isinstance(item, dict):
            continue
        qualification = " ".join(filter(None, [
            _text(item.get("degree")), _text(item.get("fieldOfStudy"))]))
        period = _text(item.get("period"))
        if not period:
            start = _text((item.get("startDate") or {}).get("text") if isinstance(item.get("startDate"), dict) else item.get("startDate"))
            end = _text((item.get("endDate") or {}).get("text") if isinstance(item.get("endDate"), dict) else item.get("endDate"))
            period = " - ".join(filter(None, [start, end]))
        education.append({
            "institution": _text(item.get("schoolName")),
            "qualification": qualification,
            "years": period,
        })

    skills = []
    for item in (row.get("skills") or []):
        if isinstance(item, dict):
            skills.append(_text(item.get("name") or item.get("title")))
        else:
            skills.append(_text(item))
    if not skills and row.get("topSkills"):
        top = row.get("topSkills")
        if isinstance(top, list):
            skills = [_text(s) for s in top if s]
        elif isinstance(top, str):
            skills = [_text(s) for s in re.split(r"[•·,\n]+", top) if _text(s)]
    skills = [s for s in skills if s]

    certifications = []
    for item in (row.get("certifications") or []):
        if not isinstance(item, dict):
            continue
        cert_name = _text(item.get("title") or item.get("name"))
        if not cert_name:
            continue
        issuer = _text(item.get("issuedBy") or item.get("authority") or item.get("issuedByOrganization"))
        issued_at = _text(item.get("issuedAt") or ((item.get("issueDate") or {}).get("text") if isinstance(item.get("issueDate"), dict) else item.get("issueDate")))
        certifications.append({
            "name": cert_name,
            "issuer": issuer,
            "date": issued_at,
        })

    languages = []
    for item in (row.get("languages") or []):
        if isinstance(item, dict):
            lang_name = _text(item.get("name") or item.get("language"))
            if lang_name:
                languages.append({
                    "name": lang_name,
                    "proficiency": _text(item.get("proficiency")),
                })
        elif isinstance(item, str) and item.strip():
            languages.append({"name": item.strip(), "proficiency": ""})

    volunteering = []
    for item in (row.get("volunteering") or []):
        if not isinstance(item, dict):
            continue
        v_role = _text(item.get("role") or item.get("title") or item.get("position"))
        v_org = _text(item.get("organizationName") or item.get("companyName") or item.get("organization"))
        v_dates = _text(item.get("duration") or item.get("period"))
        if not v_dates and isinstance(item.get("endDate"), dict):
            v_dates = _text((item.get("endDate") or {}).get("text"))
        volunteering.append({
            "role": v_role,
            "organisation": v_org,
            "cause": _text(item.get("cause")),
            "dates": v_dates,
        })

    honors = []
    for item in (row.get("honorsAndAwards") or row.get("honors") or []):
        if not isinstance(item, dict):
            continue
        h_title = _text(item.get("title") or item.get("name"))
        if not h_title:
            continue
        honors.append({
            "title": h_title,
            "issuer": _text(item.get("issuedBy") or item.get("issuer")),
            "date": _text(item.get("issuedAt") or item.get("issueDate")),
        })

    publications = []
    for item in (row.get("publications") or []):
        if not isinstance(item, dict):
            continue
        pub_title = _text(item.get("title") or item.get("name"))
        if not pub_title:
            continue
        publications.append({
            "title": pub_title,
            "date_or_publisher": _text(item.get("publishedAt") or item.get("publisher")),
        })

    courses = []
    for item in (row.get("courses") or []):
        if isinstance(item, dict):
            c_title = _text(item.get("title") or item.get("name"))
            c_assoc = _text(item.get("associatedWith"))
            if c_title:
                courses.append({"title": c_title, "associated_with": c_assoc})
        elif isinstance(item, str) and item.strip():
            courses.append({"title": item.strip(), "associated_with": ""})

    return {
        "full_name": name,
        "headline": _text(row.get("headline")),
        "company": company,
        "location": location_text,
        "email": email,
        "connections": connections,
        "followers": followers,
        "open_to_work": open_to_work,
        "verified": verified,
        "summary": _text(row.get("about")),
        "experience": experience,
        "education": education,
        "skills": skills,
        "certifications": certifications,
        "languages": languages,
        "volunteering": volunteering,
        "honors": honors,
        "publications": publications,
        "courses": courses,
    }


async def fetch_linkedin_profiles(urls: list[str]) -> dict[str, dict]:
    """Scrape several LinkedIn profiles in one Apify run.

    Returns {url: {"profile": ...}} or {url: {"error": ...}}. Batching matters:
    one actor run for the whole set is cheaper and far faster than one each.
    """
    if not APIFY_TOKEN:
        return {u: {"error": NO_APIFY_MESSAGE} for u in urls}

    endpoint = f"https://api.apify.com/v2/acts/{APIFY_ACTOR}/run-sync-get-dataset-items"
    payload = {"queries": urls, "profileScraperMode": APIFY_MODE}

    try:
        async with httpx.AsyncClient(timeout=APIFY_TIMEOUT) as http:
            response = await http.post(endpoint, params={"token": APIFY_TOKEN}, json=payload)
    except httpx.TimeoutException:
        return {u: {"error": f"Apify timed out after {APIFY_TIMEOUT:.0f}s."} for u in urls}
    except httpx.HTTPError as e:
        return {u: {"error": f"Could not reach Apify: {type(e).__name__}."} for u in urls}

    if response.status_code == 402:
        return {u: {"error": "Apify credit exhausted — top up or wait for the monthly reset."}
                for u in urls}
    if response.status_code == 401:
        return {u: {"error": "Apify rejected the token (401). Check APIFY in .env."} for u in urls}
    if response.status_code >= 400:
        detail = ""
        try:
            detail = (response.json().get("error") or {}).get("message", "")
        except Exception:
            detail = response.text[:200]
        return {u: {"error": f"Apify returned HTTP {response.status_code}: {detail}"} for u in urls}

    try:
        rows = response.json()
    except ValueError:
        return {u: {"error": "Apify returned a response that was not JSON."} for u in urls}

    by_slug: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        # The actor reports errors as rows too; keep only real profiles.
        if not (row.get("firstName") or row.get("lastName") or row.get("headline")):
            continue
        for key in filter(None, [row.get("linkedinUrl"), row.get("publicIdentifier"),
                                 ((row.get("originalQuery") or {}).get("query"))]):
            by_slug[linkedin_slug(str(key))] = row

    results: dict[str, dict] = {}
    for url in urls:
        row = by_slug.get(linkedin_slug(url))
        if row:
            results[url] = {"profile": map_apify_profile(row)}
        else:
            results[url] = {"error": (
                "Apify returned no data for this profile. It may be private, "
                "deleted, or the URL may be wrong."
            )}
    return results


async def fetch_page_text(url: str, http: httpx.AsyncClient) -> str:
    """Download a URL and reduce it to readable text for the model."""
    host = host_of(url)
    for blocked, reason in BLOCKED_HOSTS.items():
        if host == blocked or host.endswith("." + blocked):
            raise FetchError(reason)

    try:
        response = await http.get(url, headers=HTTP_HEADERS, follow_redirects=True)
    except httpx.TimeoutException:
        raise FetchError(f"Timed out after {FETCH_TIMEOUT:.0f}s.")
    except httpx.HTTPError as e:
        raise FetchError(f"Could not reach the page: {type(e).__name__}.")

    if response.status_code == 999:
        raise FetchError("The site returned HTTP 999, an anti-bot block.")
    if response.status_code == 403:
        raise FetchError("The site refused the request (HTTP 403) — likely bot protection.")
    if response.status_code == 404:
        raise FetchError("Page not found (HTTP 404). Check the URL.")
    if response.status_code >= 400:
        raise FetchError(f"The site returned HTTP {response.status_code}.")

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        raise FetchError(f"Not a readable web page (content type: {content_type or 'unknown'}).")

    soup = BeautifulSoup(response.text, "lxml")
    # Chrome and boilerplate add noise that pushes real content out of the prompt.
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header",
                     "form", "svg", "iframe"]):
        tag.decompose()

    text = " ".join(soup.get_text(" ").split())
    if len(text) < 200:
        raise FetchError(
            f"Only {len(text)} characters of text found — the page is probably "
            "rendered by JavaScript or behind a login wall."
        )
    return text[:60000]


async def parse_profile(url: str, page_text: str) -> dict:
    """Ask Gemini to fill the profile schema from the page text using concise keywords."""
    prompt = (
        "You are an executive talent researcher compiling a concise professional dossier.\n"
        "From the web page text below, extract a structured professional profile.\n\n"
        "Rules:\n"
        "- Extract details strictly as words and concise terms. NEVER write sentences or paragraphs.\n"
        "- Do not include student or personal projects. This dossier is strictly for professional corporate use.\n"
        "- For experience roles, capture title, company, dates, location, and key skills.\n"
        "- If a field is absent, use an empty string or empty list.\n\n"
        f"Page URL: {url}\n\nPage text:\n\"\"\"{page_text}\"\"\""
    )

    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=PROFILE_SCHEMA,
    )

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config=config,
            )
            break
        except genai_errors.APIError as e:
            if e.code not in RETRYABLE_CODES or attempt == MAX_ATTEMPTS - 1:
                raise FetchError(f"Gemini API error {e.code}: {e.message}")
            # Rate-limit errors carry the real wait ("Please retry in 25.2s").
            # Plain exponential backoff gives up far too early on the free tier.
            hinted = re.search(r"retry in ([\d.]+)s", str(e.message or ""), re.IGNORECASE)
            delay = min(float(hinted.group(1)) + 1, 60) if hinted else 1.5 * (2 ** attempt)
            await asyncio.sleep(delay)
    else:  # pragma: no cover - loop always breaks or raises
        raise FetchError("Gemini did not return a response.")

    import json
    try:
        data = json.loads(response.text)
    except (TypeError, ValueError) as e:
        raise FetchError(f"Gemini returned unreadable JSON: {e}")

    if not isinstance(data, dict):
        raise FetchError("Gemini returned an unexpected shape.")

    data.setdefault("certifications", [])
    data.setdefault("languages", [])
    data.setdefault("volunteering", [])
    data.setdefault("honors", [])
    data.setdefault("publications", [])
    data.setdefault("courses", [])
    data.setdefault("email", "")
    data.setdefault("connections", "")
    data.setdefault("followers", "")
    data.setdefault("open_to_work", False)
    data.setdefault("verified", False)
    return data


def build_pdf(url: str, profile: dict, path: str) -> None:
    """Render the structured profile as a concise, professional executive PDF dossier."""
    from xml.sax.saxutils import escape
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors

    def esc(value) -> str:
        return escape(str(value or "")).replace("\n", "<br/>")

    styles = getSampleStyleSheet()
    name_style = ParagraphStyle('Name', parent=styles['Heading1'], fontSize=22, leading=26,
                                textColor=colors.HexColor('#0a66c2'), spaceAfter=2)
    headline_style = ParagraphStyle('Headline', parent=styles['BodyText'], fontSize=11, leading=15,
                                    textColor=colors.HexColor('#475569'), spaceAfter=8)
    sec_style = ParagraphStyle('Sec', parent=styles['Heading2'], fontSize=12, leading=15,
                               textColor=colors.HexColor('#0f172a'), spaceBefore=12, spaceAfter=3)
    role_style = ParagraphStyle('Role', parent=styles['BodyText'], fontSize=10, leading=13,
                                textColor=colors.HexColor('#0f172a'), spaceBefore=4)
    meta_style = ParagraphStyle('Meta', parent=styles['BodyText'], fontSize=8.5, leading=11,
                                textColor=colors.HexColor('#64748b'))
    body_style = ParagraphStyle('Body', parent=styles['BodyText'], fontSize=9, leading=13,
                                textColor=colors.HexColor('#334155'))
    badge_style = ParagraphStyle('Badge', parent=styles['BodyText'], fontSize=8.5, leading=11,
                                 textColor=colors.HexColor('#057642'), spaceAfter=4)

    story = []
    story.append(Paragraph(esc(profile.get("full_name") or "Unknown"), name_style))

    # Badges (Open to Work, Verified)
    badges = []
    if profile.get("open_to_work"):
        badges.append("<font color='#057642'><b>● OPEN TO WORK</b></font>")
    if profile.get("verified"):
        badges.append("<font color='#0a66c2'><b>✔ VERIFIED PROFILE</b></font>")
    if badges:
        story.append(Paragraph(" &nbsp;·&nbsp; ".join(badges), badge_style))

    if profile.get("headline"):
        story.append(Paragraph(esc(profile["headline"]), headline_style))

    # Facts & Contact Box
    facts = []
    if profile.get("company"):
        facts.append(("Company", profile["company"]))
    if profile.get("location"):
        facts.append(("Location", profile["location"]))
    if profile.get("email"):
        facts.append(("Email", profile["email"]))

    network_parts = []
    if profile.get("connections"):
        network_parts.append(f"{profile['connections']} connections")
    if profile.get("followers"):
        network_parts.append(f"{profile['followers']} followers")
    if network_parts:
        facts.append(("Network", " · ".join(network_parts)))

    facts.append(("Source", url))

    table = Table(
        [[Paragraph(f"<b>{k}</b>", meta_style), Paragraph(esc(v), meta_style)] for k, v in facts],
        colWidths=[70, None], hAlign='LEFT',
    )
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(table)

    # Executive Overview (short phrase only, no multi-sentence paragraphs)
    if profile.get("summary"):
        first_line = profile["summary"].strip().split("\n")[0].strip()
        if first_line and len(first_line) <= 220:
            story.append(Paragraph("Executive Overview", sec_style))
            story.append(Paragraph(esc(first_line), body_style))

    # Professional Experience (strictly roles, dates, locations, key skill words — no narrative paragraphs)
    experience = [e for e in (profile.get("experience") or []) if isinstance(e, dict)]
    if experience:
        story.append(Paragraph("Professional Experience", sec_style))
        for role in experience:
            title_line = f"<b>{esc(role.get('title'))}</b>"
            if role.get('organisation'):
                title_line += f" — {esc(role['organisation'])}"
            block = [Paragraph(title_line, role_style)]
            meta_items = [role.get("dates"), role.get("location"), role.get("employment_type")]
            meta = " · ".join(filter(None, [m for m in meta_items if m]))
            if meta:
                block.append(Paragraph(esc(meta), meta_style))
            role_skills = [esc(s) for s in (role.get("skills") or []) if s]
            if role_skills:
                block.append(Paragraph(f"<b>Key competencies:</b> {' · '.join(role_skills[:8])}", meta_style))
            story.append(KeepTogether(block))

    # Education (institution, qualification, years — no paragraphs)
    education = [e for e in (profile.get("education") or []) if isinstance(e, dict)]
    if education:
        story.append(Paragraph("Education", sec_style))
        for item in education:
            line = f"<b>{esc(item.get('institution'))}</b>"
            if item.get("qualification"):
                line += f" — {esc(item['qualification'])}"
            block = [Paragraph(line, role_style)]
            if item.get("years"):
                block.append(Paragraph(esc(item["years"]), meta_style))
            story.append(KeepTogether(block))

    # Licenses & Certifications
    certifications = [c for c in (profile.get("certifications") or []) if isinstance(c, dict)]
    if certifications:
        story.append(Paragraph("Licenses & Certifications", sec_style))
        for cert in certifications:
            line = f"<b>{esc(cert.get('name'))}</b>"
            if cert.get("issuer"):
                line += f" — {esc(cert['issuer'])}"
            if cert.get("date"):
                line += f" ({esc(cert['date'])})"
            story.append(Paragraph(line, role_style))

    # Core Skills & Competencies (concise words)
    skills = [str(s) for s in (profile.get("skills") or []) if s]
    if skills:
        story.append(Paragraph("Core Skills & Competencies", sec_style))
        story.append(Spacer(1, 3))
        story.append(Paragraph(esc(" · ".join(skills)), body_style))

    # Languages
    languages = [l for l in (profile.get("languages") or []) if isinstance(l, dict)]
    if languages:
        story.append(Paragraph("Languages", sec_style))
        lang_items = []
        for lang in languages:
            name = lang.get("name")
            if not name:
                continue
            prof = lang.get("proficiency")
            lang_items.append(f"<b>{esc(name)}</b> ({esc(prof)})" if prof else f"<b>{esc(name)}</b>")
        if lang_items:
            story.append(Spacer(1, 3))
            story.append(Paragraph(" · ".join(lang_items), body_style))

    # Honors & Awards (words/title only — no paragraphs)
    honors = [h for h in (profile.get("honors") or []) if isinstance(h, dict)]
    if honors:
        story.append(Paragraph("Honors & Awards", sec_style))
        for item in honors:
            line = f"<b>{esc(item.get('title'))}</b>"
            if item.get("issuer"):
                line += f" — {esc(item['issuer'])}"
            if item.get("date"):
                line += f" ({esc(item['date'])})"
            story.append(Paragraph(line, role_style))

    # Volunteering (concise single line — no paragraphs)
    volunteering = [v for v in (profile.get("volunteering") or []) if isinstance(v, dict)]
    if volunteering:
        story.append(Paragraph("Volunteering Experience", sec_style))
        for item in volunteering:
            line = f"<b>{esc(item.get('role'))}</b>"
            if item.get("organisation"):
                line += f" — {esc(item['organisation'])}"
            meta_items = [item.get("cause"), item.get("dates")]
            meta = " · ".join(filter(None, [m for m in meta_items if m]))
            if meta:
                line += f" ({esc(meta)})"
            story.append(Paragraph(line, role_style))

    # Publications (title + publisher — no paragraphs)
    publications = [pub for pub in (profile.get("publications") or []) if isinstance(pub, dict)]
    if publications:
        story.append(Paragraph("Publications", sec_style))
        for pub in publications:
            line = f"<b>{esc(pub.get('title'))}</b>"
            if pub.get("date_or_publisher"):
                line += f" — {esc(pub['date_or_publisher'])}"
            story.append(Paragraph(line, role_style))

    # Courses
    courses = [c for c in (profile.get("courses") or [])]
    if courses:
        story.append(Paragraph("Courses", sec_style))
        course_titles = []
        for c in courses:
            if isinstance(c, dict):
                t = c.get("title") or c.get("name")
                assoc = c.get("associated_with")
                course_titles.append(f"{t} ({assoc})" if (t and assoc) else str(t or ""))
            else:
                course_titles.append(str(c))
        course_titles = [c for c in course_titles if c]
        if course_titles:
            story.append(Spacer(1, 3))
            story.append(Paragraph(" · ".join([esc(c) for c in course_titles]), body_style))

    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"Profile Dossier - {profile.get('full_name') or 'Unknown'}",
    )
    doc.build(story)


def render_result(url: str, profile: dict, out_dir: str, source: str) -> dict:
    """Write one profile to a PDF and describe the outcome."""
    name = profile.get("full_name") or "Profile"
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "Profile"
    filename = f"{safe}.pdf"
    path = os.path.join(out_dir, filename)
    suffix = 2
    while os.path.exists(path):
        filename = f"{safe}_{suffix}.pdf"
        path = os.path.join(out_dir, filename)
        suffix += 1

    try:
        build_pdf(url, profile, path)
    except Exception as e:
        return {"url": url, "ok": False, "error": f"PDF generation failed: {e}"}

    roles = len(profile.get("experience") or [])
    education = len(profile.get("education") or [])
    skills = len(profile.get("skills") or [])
    certifications = len(profile.get("certifications") or [])
    languages = len(profile.get("languages") or [])
    volunteering = len(profile.get("volunteering") or [])
    honors = len(profile.get("honors") or [])
    publications = len(profile.get("publications") or [])
    courses = len(profile.get("courses") or [])

    warning = None
    if roles == 0 and education == 0:
        warning = ("No experience or education details were found — "
                   "the dossier will be thin.")

    return {
        "url": url, "ok": True, "name": name, "source": source,
        "headline": profile.get("headline") or "",
        "company": profile.get("company") or "",
        "location": profile.get("location") or "",
        "email": profile.get("email") or "",
        "connections": profile.get("connections") or "",
        "followers": profile.get("followers") or "",
        "open_to_work": bool(profile.get("open_to_work")),
        "verified": bool(profile.get("verified")),
        "roles": roles,
        "education": education,
        "skills": skills,
        "certifications": certifications,
        "languages": languages,
        "volunteering": volunteering,
        "honors": honors,
        "publications": publications,
        "courses": courses,
        "warning": warning,
        "filename": filename, "path": path,
    }


async def process_url(url: str, http: httpx.AsyncClient, out_dir: str,
                      limiter: asyncio.Semaphore) -> dict:
    """Scrape, parse and render one non-LinkedIn URL. Never raises."""
    async with limiter:
        try:
            page_text = await fetch_page_text(url, http)
            profile = await parse_profile(url, page_text)
        except FetchError as e:
            return {"url": url, "ok": False, "error": str(e)}
        except Exception as e:  # defensive: one bad URL must not kill the batch
            return {"url": url, "ok": False, "error": f"Unexpected error: {type(e).__name__}: {e}"}

    return render_result(url, profile, out_dir, source="web + Gemini")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html",
        context={
            "max_urls": MAX_URLS,
            "apify_state": "configured" if APIFY_TOKEN else "not configured - set APIFY in .env",
        },
    )


@app.post("/extract")
async def extract(urls: str = Form(...)):
    candidates, seen = [], set()
    for line in urls.splitlines():
        url = normalise_url(line)
        if not url or url in seen:
            continue
        if not urlparse(url).hostname:
            continue
        seen.add(url)
        candidates.append(url)

    if not candidates:
        raise HTTPException(status_code=400, detail="No valid URLs were provided.")
    if len(candidates) > MAX_URLS:
        raise HTTPException(
            status_code=400,
            detail=f"{len(candidates)} URLs given; the limit is {MAX_URLS} per batch.",
        )

    job_id = uuid.uuid4().hex[:12]
    out_dir = os.path.join(GENERATED_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)

    linkedin_urls = [u for u in candidates if is_linkedin(u)]
    other_urls = [u for u in candidates if not is_linkedin(u)]

    limiter = asyncio.Semaphore(MAX_CONCURRENCY)
    by_url: dict[str, dict] = {}

    async def run_web():
        if not other_urls:
            return
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT) as http:
            for result in await asyncio.gather(
                *(process_url(u, http, out_dir, limiter) for u in other_urls)
            ):
                by_url[result["url"]] = result

    async def run_linkedin():
        if not linkedin_urls:
            return
        # One Apify run covers the whole LinkedIn set.
        scraped = await fetch_linkedin_profiles(linkedin_urls)
        for url in linkedin_urls:
            outcome = scraped.get(url) or {"error": "No result returned."}
            if "profile" in outcome:
                by_url[url] = render_result(url, outcome["profile"], out_dir, source="Apify")
            else:
                by_url[url] = {"url": url, "ok": False, "error": outcome["error"]}

    await asyncio.gather(run_web(), run_linkedin())

    # Preserve the order the user typed.
    results = [by_url[u] for u in candidates if u in by_url]
    successes = [r for r in results if r["ok"]]

    download = None
    if len(successes) == 1:
        JOBS[job_id] = {"path": successes[0]["path"], "filename": successes[0]["filename"]}
        download = f"/download/{job_id}"
    elif len(successes) > 1:
        zip_path = os.path.join(out_dir, f"dossiers_{job_id}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for item in successes:
                archive.write(item["path"], arcname=item["filename"])
        JOBS[job_id] = {"path": zip_path, "filename": f"dossiers_{job_id}.zip"}
        download = f"/download/{job_id}"

    return JSONResponse({
        "job_id": job_id,
        "total": len(results),
        "succeeded": len(successes),
        "failed": len(results) - len(successes),
        "download_url": download,
        "results": [
            {k: v for k, v in r.items() if k != "path"} for r in results
        ],
    })


@app.get("/download/{job_id}")
async def download(job_id: str):
    job = JOBS.get(job_id)
    if not job or not os.path.exists(job["path"]):
        raise HTTPException(status_code=404, detail="That download has expired or does not exist.")
    media = "application/zip" if job["filename"].endswith(".zip") else "application/pdf"
    return FileResponse(job["path"], media_type=media, filename=job["filename"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))