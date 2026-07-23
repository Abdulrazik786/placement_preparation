import os
import shutil
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from pypdf import PdfReader

import models
from database import engine, get_db
from auth import hash_password, verify_password, create_access_token, decode_access_token
from ai_service import analyze_resume, match_resume_to_job, tailor_resume_for_job

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


class UserOut(BaseModel):
    id: int
    email: str
    name: Optional[str]
    branch: Optional[str]
    graduation_year: Optional[int]
    cgpa: Optional[float]

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
    required_skills: List[str] = []


class JobPostingOut(BaseModel):
    id: int
    title: str
    company_name: str
    description: str
    required_skills: List[str]
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


# --- Company endpoints ---
@app.post("/companies", response_model=CompanyOut)
def create_company(company: CompanyCreate, db: Session = Depends(get_db)):
    # NOTE: no admin-only restriction yet — anyone logged in (or even not) can add a company right now.
    # We'll lock this down to admin-only once we add role checks; fine for local testing for now.
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


# --- Eligibility checker ---
def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


# --- Resume endpoints ---
@app.post("/resumes", response_model=ResumeOut)
def upload_resume(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now")

    # Save the file to disk with a unique name (user id + original filename, to avoid clashes)
    saved_filename = f"user{current_user.id}_{file.filename}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text right away so it's ready for the AI analysis step later
    try:
        extracted_text = extract_text_from_pdf(saved_path)
    except Exception:
        extracted_text = None  # if extraction fails, we still keep the file; text can be retried later

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
    # NOTE: same as /companies, no admin-only restriction yet - fine for local testing
    new_job = models.JobPosting(
        title=job.title,
        company_name=job.company_name,
        description=job.description,
        required_skills=job.required_skills,
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