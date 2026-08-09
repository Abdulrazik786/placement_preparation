import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
import httpx

load_dotenv()  # reads .env and loads GEMINI_API_KEY into the environment

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Using the auto-updating "latest" alias so Google can point this at whatever their
# current recommended Flash model is, without us hitting deprecation errors again later.
# As of writing, this resolves to gemini-3.5-flash.
MODEL_NAME = "gemini-flash-latest"

# Errors worth retrying: rate limits (429) and transient network issues (dropped connections,
# timeouts). These are usually momentary - a brief wait and retry resolves them without the
# user ever seeing an error. Anything else (bad request, auth failure, etc.) is not retried.
RETRYABLE_NETWORK_ERRORS = (
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
)


def _generate_with_retry(contents: str, max_retries: int = 3, initial_delay: float = 5.0):
    """
    Wraps the Gemini call with automatic retry-with-backoff for rate limit (429) errors and
    transient network issues (e.g. the connection getting dropped mid-request). Free tier allows
    only 5 requests/minute, and network hiccups happen occasionally - instead of crashing with a
    500 error, we wait and retry a couple of times before giving up.
    """
    delay = initial_delay
    last_error = None

    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=MODEL_NAME, contents=contents)
        except genai_errors.ClientError as e:
            is_rate_limit = getattr(e, "status_code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e)
            last_error = e
            if is_rate_limit and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # back off progressively longer each retry (5s, 10s, 20s...)
                continue
            raise
        except RETRYABLE_NETWORK_ERRORS as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise ValueError(f"AI service network error after {max_retries} retries: {last_error}")

    raise ValueError(f"AI service unavailable after {max_retries} retries: {last_error}")


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


SKILL_MATCH_PROMPT = """A student has these skills: {student_skills}

A job requires these skills: {required_skills}

For each required skill, determine if the student's skills genuinely cover it - accounting for synonyms,
abbreviations, version numbers, and close variants (e.g. "ML" covers "Machine Learning", "Python" covers
"Python 3", "JS" covers "JavaScript", "React.js" covers "React"). Do NOT match skills that are only loosely
related (e.g. "SQL" does NOT cover "MongoDB" - those are different technologies).

Return ONLY a JSON object (no markdown, no preamble, no code fences) with this exact shape:
{{
  "matching_skills": [<required skills, EXACTLY as worded in the input list, that the student's skills cover>],
  "missing_skills": [<required skills, EXACTLY as worded in the input list, that the student genuinely lacks>]
}}

Every skill from the required list must appear in exactly one of the two lists.
"""


def match_skills_semantically(student_skills: list, required_skills: list) -> dict:
    if not required_skills:
        return {"matching_skills": [], "missing_skills": []}
    if not student_skills:
        return {"matching_skills": [], "missing_skills": list(required_skills)}

    response = _generate_with_retry(
        SKILL_MATCH_PROMPT.format(
            student_skills=", ".join(student_skills),
            required_skills=", ".join(required_skills),
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


NEXT_QUESTION_PROMPT = """You are conducting a mock placement interview for a college student. Ask ONE question at a time,
mixing HR, technical, resume-based, and project-based questions naturally across the interview. Make each question feel
like a real interviewer's follow-up - if the candidate's last answer mentioned something specific, dig into it, the way
a human interviewer would, instead of jumping to an unrelated topic.

Candidate profile:
Skills: {skills}
Projects: {projects}
Career interest: {career_interest}

{job_context}

{resume_context}

Conversation so far (empty if this is the first question):
{formatted_history}

Return ONLY a JSON object (no markdown, no preamble, no code fences) with this exact shape:
{{
  "question_text": "<the next question>",
  "question_type": "<one of: hr, technical, resume, project>"
}}

If the conversation so far is empty, start with an opening question like "Tell me about yourself."
Ask no more than one question. Keep it concise, like a real interviewer would speak.
"""

INTERVIEW_EVALUATION_PROMPT = """Evaluate this completed mock placement interview transcript for a college student.

Candidate profile:
Skills: {skills}
Career interest: {career_interest}

{job_context}

Full transcript (question and answer pairs):
{formatted_history}

Return ONLY a JSON object (no markdown, no preamble, no code fences) with this exact shape:
{{
  "overall_score": <integer 0-100>,
  "technical_score": <integer 0-100>,
  "resume_score": <integer 0-100>,
  "project_score": <integer 0-100>,
  "communication_score": <integer 0-100>,
  "strong_areas": [<up to 5 short strings>],
  "needs_preparation": [<up to 5 short strings>],
  "question_feedback": [
    {{"question": "<question>", "answer": "<candidate's answer>", "feedback": "<1-2 sentence feedback on this specific answer>"}}
  ],
  "summary": "<3-4 sentence overall assessment>"
}}

Include one question_feedback entry per question asked in the transcript.
"""


def _format_history(history: list) -> str:
    if not history:
        return "(none yet - this is the start of the interview)"
    lines = []
    for turn in history:
        lines.append(f"Q ({turn['question_type']}): {turn['question']}")
        if turn.get("answer"):
            lines.append(f"A: {turn['answer']}")
    return "\n".join(lines)


def generate_next_interview_question(
    skills: list, projects: list, career_interest: str, job_context: str, resume_context: str, history: list
) -> dict:
    response = _generate_with_retry(
        NEXT_QUESTION_PROMPT.format(
            skills=", ".join(skills) if skills else "not specified",
            projects=json.dumps(projects) if projects else "not specified",
            career_interest=career_interest or "not specified",
            job_context=job_context,
            resume_context=resume_context,
            formatted_history=_format_history(history),
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


def evaluate_interview(skills: list, career_interest: str, job_context: str, history: list) -> dict:
    response = _generate_with_retry(
        INTERVIEW_EVALUATION_PROMPT.format(
            skills=", ".join(skills) if skills else "not specified",
            career_interest=career_interest or "not specified",
            job_context=job_context,
            formatted_history=_format_history(history),
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


RESUME_GENERATION_PROMPT = """You are an expert resume writer creating a brand-new ATS-friendly resume for a college student,
built entirely from their profile data below. Do NOT invent any experience, skills, or achievements not present
in the profile - only organize, phrase, and present what's genuinely there in the strongest possible way.

Student profile:
{profile}

Target job (if any):
{job_context}

Write a complete, well-structured resume in plain text with clear sections: Summary, Skills, Projects, Internships
(if any), Certifications (if any), Education. Use strong action verbs and quantify achievements only where the
profile data already implies a number or measurable outcome - don't fabricate metrics.

Return ONLY a JSON object (no markdown, no preamble, no code fences) with this exact shape:
{{
  "resume_text": "<the complete resume as plain text with section headers>"
}}
"""


ROADMAP_PROMPT = """Create a placement preparation roadmap for a college student.

Student profile:
{profile}

Target job (if any):
{job_context}

Skills the student is missing for this target (if any): {missing_skills}

Return ONLY a JSON object (no markdown, no preamble, no code fences) with this exact shape:
{{
  "title": "<short roadmap title, e.g. 'ML Engineer Placement Prep - 6 Week Plan'>",
  "weeks": [
    {{
      "week_number": <int starting at 1>,
      "focus_area": "<short theme for the week, e.g. 'DSA Fundamentals' or 'Docker & Deployment'>",
      "tasks": [<3-5 short, concrete task strings for that week>]
    }}
  ]
}}

Create 4-6 weeks. Prioritize the student's missing skills early, and build toward mock interview readiness
and resume/aptitude polish in later weeks. Be concrete and realistic for a student with limited daily study time.
"""

DAILY_PLAN_PROMPT = """Create TODAY's study plan for a college student preparing for placements, adjusted based on
their recent performance.

Student profile:
{profile}

Current roadmap context (if any): {roadmap_context}

Recent performance signals:
{performance_summary}

Yesterday's plan and what they completed (if available):
{yesterday_summary}

Return ONLY a JSON object (no markdown, no preamble, no code fences) with this exact shape:
{{
  "tasks": [
    {{"title": "<short task title>", "task_type": "<one of: aptitude, coding, resume, interview, learning, other>", "duration_mins": <int>}}
  ],
  "performance_note": "<1-2 sentence note explaining how today's plan was adjusted based on recent performance, written directly to the student, encouraging tone>"
}}

Include 3-6 tasks. If performance signals show a weak area (e.g. low interview score, incomplete tasks yesterday),
adjust today's plan to reinforce that area rather than just repeating the roadmap verbatim. If yesterday's tasks
were mostly completed and scores are trending up, slightly increase difficulty or introduce a new topic.
"""


def generate_roadmap(profile: dict, job_context: str, missing_skills: list) -> dict:
    response = _generate_with_retry(
        ROADMAP_PROMPT.format(
            profile=json.dumps(profile),
            job_context=job_context or "No specific job selected - general software/tech placement prep.",
            missing_skills=", ".join(missing_skills) if missing_skills else "none identified",
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


def generate_daily_plan(profile: dict, roadmap_context: str, performance_summary: str, yesterday_summary: str) -> dict:
    response = _generate_with_retry(
        DAILY_PLAN_PROMPT.format(
            profile=json.dumps(profile),
            roadmap_context=roadmap_context or "No roadmap generated yet - use general placement prep priorities.",
            performance_summary=performance_summary or "No performance history yet.",
            yesterday_summary=yesterday_summary or "No previous day's plan available.",
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


def generate_resume_from_profile(profile: dict, job_context: str) -> dict:
    response = _generate_with_retry(
        RESUME_GENERATION_PROMPT.format(
            profile=json.dumps(profile),
            job_context=job_context or "No specific job selected - write a general software/tech resume.",
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


def generate_skill_prep_batch(skills: list, role: str) -> list:
    """Generates prep content for ALL missing skills in a single AI call, to stay within free-tier rate limits."""
    if not skills:
        return []

    response = _generate_with_retry(
        SKILL_PREP_PROMPT.format(role=role or "the target role", skills_list=", ".join(skills)),
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

    response = _generate_with_retry(
        JD_EXTRACTION_PROMPT.format(description=description[:4000]),
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

    response = _generate_with_retry(
        RESUME_TAILOR_PROMPT.format(
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

    response = _generate_with_retry(
        JOB_MATCH_PROMPT.format(
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

    # truncate to ~8000 chars to keep prompt size reasonable; resumes are short documents anyway
    response = _generate_with_retry(
        ATS_ANALYSIS_PROMPT.format(resume_text=resume_text[:8000]),
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