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


JOB_MATCH_PROMPT = """You are an ATS matching engine comparing a candidate's resume against a specific job description.

Return ONLY a JSON object (no markdown, no preamble, no code fences) with this exact shape:
{{
  "match_score": <integer 0-100, how well the resume matches THIS job specifically>,
  "matching_skills": [<skills/experience from the resume that align with this job>],
  "missing_skills": [<required skills/experience from the job description NOT found in the resume>],
  "recommendation": "<2-3 sentence assessment of fit, and what to add/change to improve the match>"
}}

Job description:
---
{job_description}
---

Resume text:
---
{resume_text}
---
"""


RESUME_TAILOR_PROMPT = """You are an expert resume writer helping a college student tailor their resume for a specific job.

Rewrite and improve the resume below so it better fits the target job description. Keep it truthful — do NOT invent
experience, skills, projects, or qualifications the candidate doesn't have. Instead: reorder content to prioritize
relevant experience, rephrase bullet points to use language/keywords from the job description where honestly applicable,
tighten weak phrasing, and quantify achievements where the original already implies a metric.

Return ONLY a JSON object (no markdown, no preamble, no code fences) with this exact shape:
{{
  "tailored_resume_text": "<the full rewritten resume as plain text, preserving section structure like Education, Skills, Projects, Certifications>",
  "changes_summary": [<up to 6 short strings describing what was changed and why, e.g. 'Moved Python projects above certifications since the role prioritizes technical work'>]
}}

Job description (target role):
---
{job_description}
---

Original resume text:
---
{resume_text}
---
"""


JD_EXTRACTION_PROMPT = """Extract structured information from this job description.

Return ONLY a JSON object (no markdown, no preamble, no code fences) with this exact shape:
{{
  "role": "<normalized job title/role, e.g. 'Machine Learning Engineer'>",
  "required_skills": [<list of specific technical skills/tools/technologies mentioned or clearly implied, e.g. 'Python', 'Docker', 'AWS'>],
  "experience_summary": "<1 sentence summary of the kind of experience expected, e.g. 'ML model development and deployment'>"
}}

Job description:
---
{description}
---
"""


SKILL_PREP_PROMPT = """A college student is preparing for a "{role}" job. They are missing these skills: {skills_list}.

For EACH missing skill, provide prep guidance. Return ONLY a JSON object (no markdown, no preamble, no code fences)
with this exact shape:
{{
  "prep_topics": [
    {{
      "skill": "<the skill name, exactly as given>",
      "why_needed": "<1 sentence on why this skill matters for this specific role>",
      "key_concepts": [<up to 5 short strings naming core concepts/subtopics to learn>]
    }}
  ]
}}

Include one object per skill listed, in the same order.
"""


def generate_skill_prep_batch(skills: list, role: str) -> list:
    """Generates prep content for ALL missing skills in a single AI call, to stay within free-tier rate limits."""
    if not skills:
        return []

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=SKILL_PREP_PROMPT.format(role=role or "the target role", skills_list=", ".join(skills)),
    )
    raw_text = (response.text or "").strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        parsed = json.loads(raw_text)
        return parsed.get("prep_topics", [])
    except json.JSONDecodeError:
        raise ValueError(f"AI response was not valid JSON: {raw_text[:200]}")


def extract_job_requirements(description: str) -> dict:
    if not description or not description.strip():
        raise ValueError("Job description is empty, cannot extract requirements")

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=JD_EXTRACTION_PROMPT.format(description=description[:4000]),
    )
    raw_text = (response.text or "").strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f"AI response was not valid JSON: {raw_text[:200]}")


def tailor_resume_for_job(resume_text: str, job_description: str) -> dict:
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty, cannot tailor")
    if not job_description or not job_description.strip():
        raise ValueError("Job description is empty, cannot tailor")

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=RESUME_TAILOR_PROMPT.format(
            job_description=job_description[:4000],
            resume_text=resume_text[:8000],
        ),
    )

    raw_text = (response.text or "").strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f"AI response was not valid JSON: {raw_text[:200]}")


def match_resume_to_job(resume_text: str, job_description: str) -> dict:
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty, cannot match")
    if not job_description or not job_description.strip():
        raise ValueError("Job description is empty, cannot match")

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=JOB_MATCH_PROMPT.format(
            job_description=job_description[:4000],
            resume_text=resume_text[:8000],
        ),
    )

    raw_text = (response.text or "").strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f"AI response was not valid JSON: {raw_text[:200]}")


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