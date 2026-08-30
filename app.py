import os
import re
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI(title="LinkHarvest Enterprise AI Cloud Portal")
templates = Jinja2Templates(directory="templates")

# Initialize and configure the unified Google GenAI client layer
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

os.makedirs("generated", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def serve_portal_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/extract")
async def execute_cloud_ai_pipeline(profile_url: str = Form(...), raw_dump: str = Form(...)):
    if not raw_dump or len(raw_dump.strip()) < 100:
        raise HTTPException(status_code=400, detail="The pasted raw profile text block is too short to parse securely.")

    try:
        # THE DYNAMIC LLM INTERPRETATION MATRIX LAYER
        ai_prompt = f"""
        You are an elite corporate recruitment researcher. Analyze the following raw screen text dump of a professional profile page and extract structured values. 
        Ensure information is split accurately. Clean up any boilerplate code artifacts.
        
        Target URL context: {profile_url}
        Raw page stream input text dump:
        \"\"\"{raw_dump}\"\"\"
        
        Return the parsed results EXACTLY in this clear text-block format without markdown styling tags:
        Full Name: [Name here]
        Headline: [Headline phrase here]
        Company: [Current company name here]
        Location: [Geographic location here]
        Experience: [Top 3 roles with dates and summary description blocks here]
        Education: [Academic colleges, degrees, and years here]
        Skills: [List of top core professional skills here]
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=ai_prompt,
        )
        ai_output = response.text

        # REGEX DATA MAPPING
        def fetch_field(pattern, text):
            match = re.search(pattern, text, re.IGNORECASE)
            return match.group(1).strip() if match else "Not Listed"

        name = fetch_field(r"Full Name:\s*(.*)", ai_output)
        headline = fetch_field(r"Headline:\s*(.*)", ai_output)
        company = fetch_field(r"Company:\s*(.*)", ai_output)
        location = fetch_field(r"Location:\s*(.*)", ai_output)
        experience = re.search(r"Experience:\s*([\s\S]*?)(?=Education:|$)", ai_output, re.IGNORECASE).group(1).strip() if re.search(r"Experience:\s*([\s\S]*?)(?=Education:|$)", ai_output, re.IGNORECASE) else "Not Listed"
        education = re.search(r"Education:\s*([\s\S]*?)(?=Skills:|$)", ai_output, re.IGNORECASE).group(1).strip() if re.search(r"Education:\s*([\s\S]*?)(?=Skills:|$)", ai_output, re.IGNORECASE) else "Not Listed"
        skills = re.search(r"Skills:\s*([\s\S]*)", ai_output, re.IGNORECASE).group(1).strip() if re.search(r"Skills:\s*([\s\S]*)", ai_output, re.IGNORECASE) else "Not Listed"

        # REPORTLAB PDF COMPILER ENGINE
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        clean_name = "".join([c for c in name if c.isalnum()]).strip() or "Candidate"
        pdf_path = f"generated/CloudReport_{clean_name}.pdf"
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#0a66c2'), spaceAfter=12)
        sec_style = ParagraphStyle('Sec', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#1e293b'), spaceBefore=12, spaceAfter=6)
        body_style = ParagraphStyle('Body', parent=styles['BodyText'], fontSize=9.5, textColor=colors.HexColor('#334155'), leading=13)
        
        story.append(Paragraph(f"📄 Executive Profile Dossier: {name}", title_style))
        story.append(Paragraph(f"<b>System Target Tracking Context:</b> {profile_url}", body_style))
        story.append(Spacer(1, 10))
        
        meta_table = [
            [Paragraph("<b>Current Role:</b>", body_style), Paragraph(headline, body_style)],
            [Paragraph("<b>Active Organization:</b>", body_style), Paragraph(company, body_style)],
            [Paragraph("<b>Geographic Location:</b>", body_style), Paragraph(location, body_style)]
        ]
        
        t = Table(meta_table, colWidths=[120, 380])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(t)
        
        story.append(Paragraph("💼 Professional Career Matrix History", sec_style))
        story.append(Paragraph(experience.replace('\n', '<br/>'), body_style))
        
        story.append(Paragraph("🎓 Academic Background & Credentials", sec_style))
        story.append(Paragraph(education.replace('\n', '<br/>'), body_style))
        
        story.append(Paragraph("🛠️ Core Technical Skills & Keyword Attributes", sec_style))
        story.append(Paragraph(skills, body_style))
        
        doc.build(story)
        return FileResponse(pdf_path, media_type='application/pdf', filename=os.path.basename(pdf_path))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Cloud Processing Layer Exception: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
