
import os
import re
import json
import asyncio
import zipfile

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from google import genai


# Load private environmental configuration vault variables securely
load_dotenv()

app = FastAPI(title="LinkHarvest Enterprise AI Web Framework")
templates = Jinja2Templates(directory="templates")


# Initialize and configure the new unified Google GenAI client layer
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

COOKIES_FILE = os.path.join(os.getcwd(), "cookies.json")
os.makedirs("generated", exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def serve_portal_dashboard(request: Request):
    # Fixed parameter syntax format matching modern FastAPI framework requirements
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/extract")
async def execute_bulk_ai_pipeline(profile_urls: str = Form(...)):
    # Split text string by line breaks and remove blank row blocks safely
    urls = [
        url.strip()
        for url in profile_urls.split("\n")
        if url.strip() and "linkedin.com" in url
    ]

    if not urls:
        raise HTTPException(
            status_code=400,
            detail="No valid target URLs provided in the queue list."
        )

    if not os.path.exists(COOKIES_FILE):
        raise HTTPException(
            status_code=500,
            detail=(
                "Configuration Error: 'cookies.json' file is missing "
                "in the project folder directory path."
            )
        )

    generated_pdfs = []

    # 1. PLAYWRIGHT AUTOMATED COOKIE-INJECTED BACKGROUND COLLECTOR
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        )

        # Inject session tokens with an absolute sanitation
        # reconstruction map to prevent sameSite crashes
        try:
            with open(COOKIES_FILE, "r") as f:
                raw_cookies = json.load(f)
                sanitized_cookies = []

                # RECONSTRUCTION LOOP:
                # Builds a fresh, verified dictionary track
                for c in raw_cookies:
                    clean_cookie = {
                        "name": c.get("name"),
                        "value": c.get("value"),
                        "domain": c.get("domain"),
                        "path": c.get("path", "/")
                    }

                    # Force strict string capitalization values
                    # matching Playwright's core schema rules
                    s_site = str(
                        c.get("sameSite", "")
                    ).strip().lower()

                    if s_site == "lax":
                        clean_cookie["sameSite"] = "Lax"

                    elif s_site == "strict":
                        clean_cookie["sameSite"] = "Strict"

                    elif s_site == "none":
                        clean_cookie["sameSite"] = "None"

                    else:
                        # Fallback setting to completely bypass
                        # expected Strict/Lax/None syntax crashes
                        clean_cookie["sameSite"] = "None"

                    # Append optional tracking variables safely
                    if (
                        "expirationDate" in c
                        and c["expirationDate"] is not None
                    ):
                        clean_cookie["expires"] = int(
                            c["expirationDate"]
                        )

                    if "secure" in c:
                        clean_cookie["secure"] = bool(
                            c["secure"]
                        )

                    if "httpOnly" in c:
                        clean_cookie["httpOnly"] = bool(
                            c["httpOnly"]
                        )

                    sanitized_cookies.append(clean_cookie)

                await context.add_cookies(sanitized_cookies)

            print(
                "[🟢 SUCCESS] Active burner session cookies "
                "perfectly reconstructed and injected smoothly."
            )

        except Exception as e:
            await browser.close()

            raise HTTPException(
                status_code=500,
                detail=(
                    "Failed to compile or parse cookies configuration "
                    f"array safely: {str(e)}"
                )
            )

        page = await context.new_page()

        # Process every target URL
        for target_url in urls:
            try:
                await page.goto(
                    target_url,
                    timeout=45000,
                    wait_until="load"
                )

                await page.wait_for_load_state("networkidle")

                # Safe buffer delay for elements parsing
                await asyncio.sleep(6)

                raw_page_text = await page.evaluate(
                    "() => document.body.innerText"
                )

                if (
                    not raw_page_text
                    or len(raw_page_text.strip()) < 100
                    or "login" in page.url
                ):
                    print(
                        f"[⚠️ WARNING] Skipped profile link "
                        f"{target_url} because session hit a "
                        "redirect boundary wall."
                    )
                    continue

                # 2. QUERY GENERATIVE LLM BRAIN
                ai_prompt = f"""
You are an elite corporate recruitment researcher.
Analyze the following raw screen text dump of a professional
profile page and extract structured values.

Ensure information is split accurately.
Clean up any boilerplate code artifacts.

Target URL context: {target_url}

Raw page stream input text dump:
\"\"\"{raw_page_text}\"\"\"

Return the parsed results EXACTLY in this clear text-block
format without markdown styling tags:

Full Name: [Name here]
Headline: [Headline phrase here]
Company: [Current company name here]
Location: [Geographic location here]
Experience: [Top 3 roles with dates and summary description blocks here]
Education: [Academic colleges, degrees, and years here]
Skills: [List of top core professional skills here]
"""

                # Queries the reliable steady-state production
                # build tracking layer
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=ai_prompt,
                )

                ai_output = response.text

                # 3. REGEX DATA MAPPING
                def fetch_field(pattern, text):
                    match = re.search(
                        pattern,
                        text,
                        re.IGNORECASE
                    )

                    return (
                        match.group(1).strip()
                        if match
                        else "Not Listed"
                    )

                name = fetch_field(
                    r"Full Name:\s*(.*)",
                    ai_output
                )

                headline = fetch_field(
                    r"Headline:\s*(.*)",
                    ai_output
                )

                company = fetch_field(
                    r"Company:\s*(.*)",
                    ai_output
                )

                location = fetch_field(
                    r"Location:\s*(.*)",
                    ai_output
                )

                experience_match = re.search(
                    r"Experience:\s*([\s\S]*?)(?=Education:|$)",
                    ai_output,
                    re.IGNORECASE
                )

                experience = (
                    experience_match.group(1).strip()
                    if experience_match
                    else "Not Listed"
                )

                education_match = re.search(
                    r"Education:\s*([\s\S]*?)(?=Skills:|$)",
                    ai_output,
                    re.IGNORECASE
                )

                education = (
                    education_match.group(1).strip()
                    if education_match
                    else "Not Listed"
                )

                skills_match = re.search(
                    r"Skills:\s*([\s\S]*)",
                    ai_output,
                    re.IGNORECASE
                )

                skills = (
                    skills_match.group(1).strip()
                    if skills_match
                    else "Not Listed"
                )

                # 4. DRAW REPORTLAB PDF DOSSIER
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import (
                    SimpleDocTemplate,
                    Paragraph,
                    Spacer,
                    Table,
                    TableStyle
                )
                from reportlab.lib.styles import (
                    getSampleStyleSheet,
                    ParagraphStyle
                )
                from reportlab.lib import colors

                clean_name = "".join(
                    c for c in name
                    if c.isalnum()
                ).strip() or "Candidate"

                pdf_path = (
                    f"generated/ProfileReport_{clean_name}.pdf"
                )

                doc = SimpleDocTemplate(
                    pdf_path,
                    pagesize=letter,
                    rightMargin=40,
                    leftMargin=40,
                    topMargin=40,
                    bottomMargin=40
                )

                story = []
                styles = getSampleStyleSheet()

                title_style = ParagraphStyle(
                    "Title",
                    parent=styles["Heading1"],
                    fontSize=20,
                    textColor=colors.HexColor("#0a66c2"),
                    spaceAfter=12
                )

                sec_style = ParagraphStyle(
                    "Sec",
                    parent=styles["Heading2"],
                    fontSize=12,
                    textColor=colors.HexColor("#1e293b"),
                    spaceBefore=12,
                    spaceAfter=6
                )

                body_style = ParagraphStyle(
                    "Body",
                    parent=styles["BodyText"],
                    fontSize=9.5,
                    textColor=colors.HexColor("#334155"),
                    leading=13
                )

                story.append(
                    Paragraph(
                        f"📄 Executive Dossier: {name}",
                        title_style
                    )
                )

                story.append(
                    Paragraph(
                        f"<b>Context URL:</b> {target_url}",
                        body_style
                    )
                )

                story.append(Spacer(1, 10))

                meta_table = [
                    [
                        Paragraph(
                            "<b>Current Role:</b>",
                            body_style
                        ),
                        Paragraph(
                            headline,
                            body_style
                        )
                    ],
                    [
                        Paragraph(
                            "<b>Organization:</b>",
                            body_style
                        ),
                        Paragraph(
                            company,
                            body_style
                        )
                    ],
                    [
                        Paragraph(
                            "<b>Location:</b>",
                            body_style
                        ),
                        Paragraph(
                            location,
                            body_style
                        )
                    ]
                ]

                # Assigned valid dimensions array parameter
                # [Labels Width, Data Width]
                t = Table(
                    meta_table,
                    colWidths=[120, 400]
                )

                t.setStyle(
                    TableStyle(
                        [
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, -1),
                                colors.HexColor("#f8fafc")
                            ),
                            (
                                "VALIGN",
                                (0, 0),
                                (-1, -1),
                                "TOP"
                            ),
                            (
                                "GRID",
                                (0, 0),
                                (-1, -1),
                                0.5,
                                colors.HexColor("#e2e8f0")
                            ),
                            (
                                "TOPPADDING",
                                (0, 0),
                                (-1, -1),
                                5
                            ),
                            (
                                "BOTTOMPADDING",
                                (0, 0),
                                (-1, -1),
                                5
                            ),
                        ]
                    )
                )

                story.append(t)

                story.append(
                    Paragraph(
                        "💼 Work Experience History",
                        sec_style
                    )
                )

                story.append(
                    Paragraph(
                        experience.replace("\n", ""),
                        body_style
                    )
                )

                story.append(
                    Paragraph(
                        "🎓 Education Background",
                        sec_style
                    )
                )

                story.append(
                    Paragraph(
                        education.replace("\n", ""),
                        body_style
                    )
                )

                story.append(
                    Paragraph(
                        "🛠️ Core Professional Skills",
                        sec_style
                    )
                )

                story.append(
                    Paragraph(
                        skills,
                        body_style
                    )
                )

                doc.build(story)

                generated_pdfs.append(pdf_path)

            except Exception as e:
                print(
                    f"[⚠️ ERROR] Skipping link {target_url} "
                    f"due to parsing exception: {str(e)}"
                )
                continue

        # Close browser resources only after all URLs
        # have been processed.
        await context.close()
        await browser.close()

    # 5. PACK GENERATED REPORTS INTO A SINGLE
    # COMPRESSED ZIP FILE
    if len(generated_pdfs) == 1:
        return FileResponse(
            generated_pdfs[0],
            media_type="application/pdf",
            filename=os.path.basename(
                generated_pdfs[0]
            )
        )

    elif len(generated_pdfs) > 1:
        zip_path = (
            "generated/LinkHarvest_BulkCampaign_Reports.zip"
        )

        with zipfile.ZipFile(
            zip_path,
            "w"
        ) as zipf:

            for pdf in generated_pdfs:
                zipf.write(
                    pdf,
                    os.path.basename(pdf)
                )

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="LinkHarvest_BulkCampaign_Reports.zip"
        )

    else:
        raise HTTPException(
            status_code=500,
            detail=(
                "Bulk campaign finished, but no profile details "
                "could be compiled safely. Double check your "
                "cookies.json file validity."
            )
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )

