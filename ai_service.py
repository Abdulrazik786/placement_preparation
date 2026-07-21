import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()  # reads .env and loads GEMINI_API_KEY into the environment

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Using the auto-updating "latest" alias so Google can point this at whatever their
# current recommended Flash model is, without us hitting deprecation errors again later.
# As of writing, this resolves to gemini-3.5-flash.
MODEL_NAME = "gemini-flash-latest"

ATS_ANALYSIS_PROMPT = """You are an ATS (Applicant Tracking System) resume reviewer for college placement preparation.

Analyze the following resume text and return ONLY a JSON object (no markdown, no preamble, no code fences) with this exact shape:
{{
  "ats_score": <integer 0-100>,
  "strengths": [<up to 4 short strings>],
  "weaknesses": [<up to 4 short strings>],
  "missing_sections": [<strings, e.g. "Projects", "Certifications" if genuinely absent>],
  "keyword_suggestions": [<up to 6 relevant technical/domain keywords the resume should consider adding, based on the target role if mentioned, otherwise general software/tech roles>],
  "formatting_issues": [<up to 3 short strings, e.g. "Uses tables which some ATS systems can't parse">],
  "summary": "<2-3 sentence overall assessment>"
}}

Resume text:
---
{resume_text}
---
"""


def analyze_resume(resume_text: str) -> dict:
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty, cannot analyze")

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=ATS_ANALYSIS_PROMPT.format(resume_text=resume_text[:8000]),
        # truncate to ~8000 chars to keep prompt size reasonable; resumes are short documents anyway
    )

    raw_text = (response.text or "").strip()

    # Model is instructed to return only JSON, but strip code fences just in case
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f"AI response was not valid JSON: {raw_text[:200]}")