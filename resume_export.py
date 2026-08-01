import io
import re

from docx import Document
from docx.shared import Pt
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# Common resume section headers we look for to apply heading styling.
# Matched case-insensitively against a line's stripped text.
KNOWN_SECTION_HEADERS = {
    "summary", "objective", "skills", "technical skills", "projects",
    "internships", "internship", "experience", "work experience",
    "certifications", "education", "achievements", "profile",
}


def _split_into_lines(resume_text: str) -> list:
    return [line.strip() for line in resume_text.split("\n")]


def _is_header(line: str) -> bool:
    if not line:
        return False
    normalized = line.strip(":").lower()
    if normalized in KNOWN_SECTION_HEADERS:
        return True
    # Fallback: short, all-caps lines are likely headers too (e.g. "SKILLS", "PROJECTS")
    return line.isupper() and 2 <= len(line.split()) <= 4


def build_docx(resume_text: str) -> io.BytesIO:
    doc = Document()

    # Slightly tighter default margins/font for a denser, resume-like look
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    for line in _split_into_lines(resume_text):
        if not line:
            doc.add_paragraph("")
            continue
        if _is_header(line):
            heading = doc.add_heading(line.upper(), level=2)
            heading.style.font.size = Pt(13)
        else:
            doc.add_paragraph(line)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer


def build_pdf(resume_text: str) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=14, spaceAfter=4)
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=12,
        spaceBefore=10, spaceAfter=4, textColor="#1a1a1a",
    )

    story = []
    for line in _split_into_lines(resume_text):
        if not line:
            story.append(Spacer(1, 6))
            continue
        # Escape characters that would break reportlab's mini-HTML parsing
        safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if _is_header(line):
            story.append(Paragraph(safe_line.upper(), heading_style))
        else:
            story.append(Paragraph(safe_line, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer