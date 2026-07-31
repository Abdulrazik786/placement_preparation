import os
import shutil
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from pypdf import PdfReader

import models
from database import engine, get_db
from auth import hash_password, verify_password, create_access_token, decode_access_token
from ai_service import (
    analyze_resume,
    match_resume_to_job,
    tailor_resume_for_job,
    extract_job_requirements,
    generate_skill_prep_batch,
    generate_next_interview_question,
    evaluate_interview,
)

# Folder where uploaded resume files get saved
UPLOAD_DIR = "uploaded_resumes"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Creates the SQLite tables based on models.py, if they don't already exist
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Placement Prep API")

# Tells FastAPI where clients should send username/password to get a token (used by /docs UI too)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# ---- Pydantic schemas (what the API accepts/returns) ----
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None
    branch: Optional[str] = None
    graduation_year: Optional[int] = None
    cgpa: Optional[float] = None


class ProjectItem(BaseModel):
    title: str
    description: str
    tech_stack: List[str] = []


class CertificationItem(BaseModel):
    name: str
    issuer: Optional[str] = None
    year: Optional[int] = None


class InternshipItem(BaseModel):
    role: str
    company: str
    duration: Optional[str] = None
    description: Optional[str] = None


class ProfileUpdate(BaseModel):
    # All optional - student can update just one part of their profile at a time
    skills: Optional[List[str]] = None
    projects: Optional[List[ProjectItem]] = None
    certifications: Optional[List[CertificationItem]] = None
    internships: Optional[List[InternshipItem]] = None
    career_interest: Optional[str] = None


class UserOut(BaseModel):
    id: int
    email: str
    name: Optional[str]
    branch: Optional[str]
    graduation_year: Optional[int]
    cgpa: Optional[float]
    skills: List[str]
    projects: List[dict]
    certifications: List[dict]
    internships: List[dict]
    career_interest: Optional[str]

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class CompanyCreate(BaseModel):
    name: str
    min_cgpa: Optional[float] = None
    max_backlogs_allowed: Optional[int] = None
    allowed_branches: List[str] = []  # e.g. ["CSE", "AIML", "IT"], empty list = all branches allowed


class CompanyOut(BaseModel):
    id: int
    name: str
    min_cgpa: Optional[float]
    max_backlogs_allowed: Optional[int]
    allowed_branches: List[str]

    class Config:
        from_attributes = True


class ResumeOut(BaseModel):
    id: int
    filename: str
    uploaded_at: datetime
    extracted_text: Optional[str]

    class Config:
        from_attributes = True


class ResumeAnalysisOut(BaseModel):
    ats_score: int
    strengths: List[str]
    weaknesses: List[str]
    missing_sections: List[str]
    keyword_suggestions: List[str]
    formatting_issues: List[str]
    summary: str


class JobPostingCreate(BaseModel):
    title: str
    company_name: str
    description: str
    required_skills: Optional[List[str]] = None  # if not given, AI extracts these from the description


class JobPostingOut(BaseModel):
    id: int
    title: str
    company_name: str
    description: str
    role: Optional[str]
    required_skills: List[str]
    experience_summary: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class JobMatchOut(BaseModel):
    match_score: int
    matching_skills: List[str]
    missing_skills: List[str]
    recommendation: str


class TailoredResumeOut(BaseModel):
    id: int
    tailored_text: str
    changes_summary: List[str]
    created_at: datetime

    class Config:
        from_attributes = True


class SkillPrepTopic(BaseModel):
    skill: str
    why_needed: str
    key_concepts: List[str]


class SkillGapOut(BaseModel):
    job_title: str
    role: Optional[str]
    matching_skills: List[str]
    missing_skills: List[str]
    prep_topics: List[SkillPrepTopic]


class InterviewStartRequest(BaseModel):
    job_id: Optional[int] = None
    resume_id: Optional[int] = None


class InterviewQuestionOut(BaseModel):
    session_id: int
    question_text: str
    question_type: str
    status: str


class InterviewAnswerRequest(BaseModel):
    answer: str
    end_interview: bool = False  # set True to end the interview instead of getting another question


class InterviewMessageOut(BaseModel):
    sender: str
    question_type: Optional[str]
    content: str

    class Config:
        from_attributes = True


class InterviewSessionOut(BaseModel):
    id: int
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    messages: List[InterviewMessageOut]

    class Config:
        from_attributes = True


class InterviewQuestionFeedback(BaseModel):
    question: str
    answer: str
    feedback: str


class InterviewEvaluationOut(BaseModel):
    overall_score: int
    technical_score: int
    resume_score: int
    project_score: int
    communication_score: int
    strong_areas: List[str]
    needs_preparation: List[str]
    question_feedback: List[InterviewQuestionFeedback]
    summary: str

    class Config:
        from_attributes = True


class EligibilityResult(BaseModel):
    company: CompanyOut
    eligible: bool
    reasons: List[str]  # explains why not eligible, empty if eligible


# ---- Auth dependency: figures out which user is making the request from their token ----
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    user_id = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    user = db.query(models.User).filter(models.User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception
    return user


# ---- Eligibility logic (kept as a plain function so it's easy to test/reuse) ----
def check_eligibility(user: models.User, company: models.Company) -> EligibilityResult:
    reasons = []

    if company.min_cgpa is not None:
        if user.cgpa is None:
            reasons.append("Your CGPA is not set on your profile")
        elif user.cgpa < company.min_cgpa:
            reasons.append(f"CGPA {user.cgpa} is below the required {company.min_cgpa}")

    if company.max_backlogs_allowed is not None and user.backlogs > company.max_backlogs_allowed:
        reasons.append(f"You have {user.backlogs} backlog(s), max allowed is {company.max_backlogs_allowed}")

    if company.allowed_branches:  # empty list means "all branches allowed", so only check if it's non-empty
        if not user.branch or user.branch.upper() not in [b.upper() for b in company.allowed_branches]:
            reasons.append(f"Branch '{user.branch}' is not in the allowed list: {company.allowed_branches}")

    return EligibilityResult(company=company, eligible=(len(reasons) == 0), reasons=reasons)


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def build_qa_history(messages: list) -> list:
    """Pairs up interviewer questions with the candidate's following answer, for feeding back into the AI."""
    history = []
    pending = None
    for m in messages:
        if m.sender == "interviewer":
            if pending:
                history.append(pending)
            pending = {"question_type": m.question_type, "question": m.content, "answer": None}
        elif m.sender == "candidate" and pending:
            pending["answer"] = m.content
    if pending:
        history.append(pending)
    return history


def build_job_context(job: Optional[models.JobPosting]) -> str:
    if not job:
        return "No specific job selected - conduct a general software/tech placement interview."
    return (
        f"Target role: {job.role or job.title} at {job.company_name}\n"
        f"Job description: {job.description[:1000]}\n"
        f"Required skills: {', '.join(job.required_skills) if job.required_skills else 'not specified'}"
    )


def build_resume_context(resume: Optional[models.Resume]) -> str:
    if not resume or not resume.extracted_text:
        return "No resume provided for this interview."
    return f"Candidate's resume:\n{resume.extracted_text[:2000]}"


# ---- Routes ----
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Placement Prep API is running"}


@app.post("/users", response_model=UserOut)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = models.User(
        email=user.email,
        hashed_password=hash_password(user.password),
        name=user.name,
        branch=user.branch,
        graduation_year=user.graduation_year,
        cgpa=user.cgpa,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/{user_id}", response_model=UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.get("/me", response_model=UserOut)
def read_current_user(current_user: models.User = Depends(get_current_user)):
    return current_user


@app.put("/me/profile", response_model=UserOut)
def update_my_profile(
    profile: ProfileUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Only overwrite fields that were actually provided - lets the student update
    # e.g. just skills, without wiping out their existing projects
    if profile.skills is not None:
        current_user.skills = profile.skills
    if profile.projects is not None:
        current_user.projects = [p.dict() for p in profile.projects]
    if profile.certifications is not None:
        current_user.certifications = [c.dict() for c in profile.certifications]
    if profile.internships is not None:
        current_user.internships = [i.dict() for i in profile.internships]
    if profile.career_interest is not None:
        current_user.career_interest = profile.career_interest

    db.commit()
    db.refresh(current_user)
    return current_user


# --- Company endpoints ---
@app.post("/companies", response_model=CompanyOut)
def create_company(company: CompanyCreate, db: Session = Depends(get_db)):
    existing = db.query(models.Company).filter(models.Company.name == company.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Company already exists")

    new_company = models.Company(
        name=company.name,
        min_cgpa=company.min_cgpa,
        max_backlogs_allowed=company.max_backlogs_allowed,
        allowed_branches=company.allowed_branches,
    )
    db.add(new_company)
    db.commit()
    db.refresh(new_company)
    return new_company


@app.get("/companies", response_model=List[CompanyOut])
def list_companies(db: Session = Depends(get_db)):
    return db.query(models.Company).all()


# --- Resume endpoints ---
@app.post("/resumes", response_model=ResumeOut)
def upload_resume(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now")

    saved_filename = f"user{current_user.id}_{file.filename}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        extracted_text = extract_text_from_pdf(saved_path)
    except Exception:
        extracted_text = None

    new_resume = models.Resume(
        user_id=current_user.id,
        filename=file.filename,
        file_path=saved_path,
        extracted_text=extracted_text,
    )
    db.add(new_resume)
    db.commit()
    db.refresh(new_resume)
    return new_resume


@app.get("/resumes", response_model=List[ResumeOut])
def list_my_resumes(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return db.query(models.Resume).filter(models.Resume.user_id == current_user.id).all()


@app.post("/resumes/{resume_id}/analyze", response_model=ResumeAnalysisOut)
def analyze_my_resume(
    resume_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    resume = (
        db.query(models.Resume)
        .filter(models.Resume.id == resume_id, models.Resume.user_id == current_user.id)
        .first()
    )
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    if not resume.extracted_text:
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from this resume (it may be a scanned image PDF)",
        )

    try:
        result = analyze_resume(resume.extracted_text)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"AI analysis failed: {str(e)}")

    return result


@app.post("/jobs", response_model=JobPostingOut)
def create_job(job: JobPostingCreate, db: Session = Depends(get_db)):
    # Always run extraction to get "role" and "experience_summary"; only use the AI's
    # required_skills if the caller didn't provide their own list explicitly.
    try:
        extracted = extract_job_requirements(job.description)
    except ValueError:
        extracted = {"role": None, "required_skills": [], "experience_summary": None}

    new_job = models.JobPosting(
        title=job.title,
        company_name=job.company_name,
        description=job.description,
        role=extracted.get("role"),
        required_skills=job.required_skills if job.required_skills is not None else extracted.get("required_skills", []),
        experience_summary=extracted.get("experience_summary"),
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)
    return new_job


@app.get("/jobs", response_model=List[JobPostingOut])
def list_jobs(db: Session = Depends(get_db)):
    return db.query(models.JobPosting).order_by(models.JobPosting.created_at.desc()).all()


@app.post("/resumes/{resume_id}/match/{job_id}", response_model=JobMatchOut)
def match_resume_to_job_posting(
    resume_id: int,
    job_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    resume = (
        db.query(models.Resume)
        .filter(models.Resume.id == resume_id, models.Resume.user_id == current_user.id)
        .first()
    )
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    if not resume.extracted_text:
        raise HTTPException(status_code=400, detail="No text could be extracted from this resume")

    job = db.query(models.JobPosting).filter(models.JobPosting.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job posting not found")

    try:
        result = match_resume_to_job(resume.extracted_text, job.description)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"AI matching failed: {str(e)}")

    return result


@app.post("/resumes/{resume_id}/tailor/{job_id}", response_model=TailoredResumeOut)
def tailor_my_resume(
    resume_id: int,
    job_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    resume = (
        db.query(models.Resume)
        .filter(models.Resume.id == resume_id, models.Resume.user_id == current_user.id)
        .first()
    )
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")
    if not resume.extracted_text:
        raise HTTPException(status_code=400, detail="No text could be extracted from this resume")

    job = db.query(models.JobPosting).filter(models.JobPosting.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job posting not found")

    try:
        result = tailor_resume_for_job(resume.extracted_text, job.description)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"AI tailoring failed: {str(e)}")

    new_tailored = models.TailoredResume(
        user_id=current_user.id,
        original_resume_id=resume.id,
        job_posting_id=job.id,
        tailored_text=result["tailored_resume_text"],
        changes_summary=result["changes_summary"],
    )
    db.add(new_tailored)
    db.commit()
    db.refresh(new_tailored)

    return TailoredResumeOut(
        id=new_tailored.id,
        tailored_text=new_tailored.tailored_text,
        changes_summary=new_tailored.changes_summary,
        created_at=new_tailored.created_at,
    )


@app.get("/tailored-resumes", response_model=List[TailoredResumeOut])
def list_my_tailored_resumes(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    results = (
        db.query(models.TailoredResume)
        .filter(models.TailoredResume.user_id == current_user.id)
        .order_by(models.TailoredResume.created_at.desc())
        .all()
    )
    return [
        TailoredResumeOut(
            id=r.id,
            tailored_text=r.tailored_text,
            changes_summary=r.changes_summary,
            created_at=r.created_at,
        )
        for r in results
    ]


@app.get("/jobs/{job_id}/skill-gap", response_model=SkillGapOut)
def get_skill_gap(
    job_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    job = db.query(models.JobPosting).filter(models.JobPosting.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job posting not found")

    student_skills_lower = {s.lower().strip() for s in (current_user.skills or [])}
    required = job.required_skills or []

    matching = [s for s in required if s.lower().strip() in student_skills_lower]
    missing = [s for s in required if s.lower().strip() not in student_skills_lower]

    prep_topics = []
    if missing:
        try:
            raw_prep_topics = generate_skill_prep_batch(missing, job.role)
            prep_topics = [
                SkillPrepTopic(
                    skill=p.get("skill", ""),
                    why_needed=p.get("why_needed", ""),
                    key_concepts=p.get("key_concepts", []),
                )
                for p in raw_prep_topics
            ]
        except ValueError:
            # if AI generation fails, return the gap without prep content rather than failing the whole request
            pass

    return SkillGapOut(
        job_title=job.title,
        role=job.role,
        matching_skills=matching,
        missing_skills=missing,
        prep_topics=prep_topics,
    )


@app.post("/interviews/start", response_model=InterviewQuestionOut)
def start_interview(
    request: InterviewStartRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    job = None
    if request.job_id is not None:
        job = db.query(models.JobPosting).filter(models.JobPosting.id == request.job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job posting not found")

    resume = None
    if request.resume_id is not None:
        resume = (
            db.query(models.Resume)
            .filter(models.Resume.id == request.resume_id, models.Resume.user_id == current_user.id)
            .first()
        )
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")

    session = models.InterviewSession(
        user_id=current_user.id,
        job_posting_id=job.id if job else None,
        resume_id=resume.id if resume else None,
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    try:
        result = generate_next_interview_question(
            skills=current_user.skills or [],
            projects=current_user.projects or [],
            career_interest=current_user.career_interest,
            job_context=build_job_context(job),
            resume_context=build_resume_context(resume),
            history=[],
        )
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"AI question generation failed: {str(e)}")

    first_message = models.InterviewMessage(
        session_id=session.id,
        sender="interviewer",
        content=result["question_text"],
        question_type=result.get("question_type"),
    )
    db.add(first_message)
    db.commit()

    return InterviewQuestionOut(
        session_id=session.id,
        question_text=result["question_text"],
        question_type=result.get("question_type", "hr"),
        status=session.status,
    )


@app.post("/interviews/{session_id}/respond", response_model=InterviewQuestionOut)
def respond_to_interview(
    session_id: int,
    answer: InterviewAnswerRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = (
        db.query(models.InterviewSession)
        .filter(models.InterviewSession.id == session_id, models.InterviewSession.user_id == current_user.id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    if session.status == "completed":
        raise HTTPException(status_code=400, detail="This interview has already ended")

    # Save the candidate's answer
    candidate_message = models.InterviewMessage(
        session_id=session.id, sender="candidate", content=answer.answer
    )
    db.add(candidate_message)
    db.commit()

    # Student explicitly chose to end the interview (the "Done" button) - no more questions generated
    if answer.end_interview:
        session.status = "completed"
        session.completed_at = func.now()
        db.commit()
        return InterviewQuestionOut(
            session_id=session.id,
            question_text="Interview ended. You can now request your evaluation.",
            question_type="hr",
            status="completed",
        )

    job = db.query(models.JobPosting).filter(models.JobPosting.id == session.job_posting_id).first() if session.job_posting_id else None
    resume = db.query(models.Resume).filter(models.Resume.id == session.resume_id).first() if session.resume_id else None

    all_messages = (
        db.query(models.InterviewMessage)
        .filter(models.InterviewMessage.session_id == session.id)
        .order_by(models.InterviewMessage.id)
        .all()
    )
    history = build_qa_history(all_messages)

    try:
        result = generate_next_interview_question(
            skills=current_user.skills or [],
            projects=current_user.projects or [],
            career_interest=current_user.career_interest,
            job_context=build_job_context(job),
            resume_context=build_resume_context(resume),
            history=history,
        )
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"AI question generation failed: {str(e)}")

    next_message = models.InterviewMessage(
        session_id=session.id,
        sender="interviewer",
        content=result["question_text"],
        question_type=result.get("question_type"),
    )
    db.add(next_message)
    db.commit()

    return InterviewQuestionOut(
        session_id=session.id,
        question_text=result["question_text"],
        question_type=result.get("question_type", "hr"),
        status=session.status,
    )


@app.post("/interviews/{session_id}/end", response_model=InterviewSessionOut)
def end_interview(
    session_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = (
        db.query(models.InterviewSession)
        .filter(models.InterviewSession.id == session_id, models.InterviewSession.user_id == current_user.id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")

    if session.status != "completed":
        session.status = "completed"
        session.completed_at = func.now()
        db.commit()
        db.refresh(session)

    return session


@app.get("/interviews", response_model=List[InterviewSessionOut])
def list_my_interviews(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return (
        db.query(models.InterviewSession)
        .filter(models.InterviewSession.user_id == current_user.id)
        .order_by(models.InterviewSession.created_at.desc())
        .all()
    )


@app.get("/interviews/{session_id}", response_model=InterviewSessionOut)
def get_interview_transcript(
    session_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = (
        db.query(models.InterviewSession)
        .filter(models.InterviewSession.id == session_id, models.InterviewSession.user_id == current_user.id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")
    return session


@app.post("/interviews/{session_id}/evaluate", response_model=InterviewEvaluationOut)
def evaluate_my_interview(
    session_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = (
        db.query(models.InterviewSession)
        .filter(models.InterviewSession.id == session_id, models.InterviewSession.user_id == current_user.id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found")

    existing_eval = (
        db.query(models.InterviewEvaluation)
        .filter(models.InterviewEvaluation.session_id == session.id)
        .first()
    )
    if existing_eval:
        return existing_eval

    job = db.query(models.JobPosting).filter(models.JobPosting.id == session.job_posting_id).first() if session.job_posting_id else None

    all_messages = (
        db.query(models.InterviewMessage)
        .filter(models.InterviewMessage.session_id == session.id)
        .order_by(models.InterviewMessage.id)
        .all()
    )
    history = build_qa_history(all_messages)
    if not history:
        raise HTTPException(status_code=400, detail="No interview questions/answers found to evaluate")

    try:
        result = evaluate_interview(
            skills=current_user.skills or [],
            career_interest=current_user.career_interest,
            job_context=build_job_context(job),
            history=history,
        )
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"AI evaluation failed: {str(e)}")

    new_eval = models.InterviewEvaluation(
        session_id=session.id,
        overall_score=result["overall_score"],
        technical_score=result["technical_score"],
        resume_score=result["resume_score"],
        project_score=result["project_score"],
        communication_score=result["communication_score"],
        strong_areas=result.get("strong_areas", []),
        needs_preparation=result.get("needs_preparation", []),
        question_feedback=result.get("question_feedback", []),
        summary=result.get("summary", ""),
    )
    db.add(new_eval)

    session.status = "completed"
    if not session.completed_at:
        session.completed_at = func.now()

    db.commit()
    db.refresh(new_eval)
    return new_eval


@app.get("/eligibility", response_model=List[EligibilityResult])
def get_my_eligibility(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    companies = db.query(models.Company).all()
    return [check_eligibility(current_user, company) for company in companies]


@app.get("/eligibility/{company_id}", response_model=EligibilityResult)
def get_my_eligibility_for_company(
    company_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    company = db.query(models.Company).filter(models.Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return check_eligibility(current_user, company)