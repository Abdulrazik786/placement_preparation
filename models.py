from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class College(Base):
    __tablename__ = "colleges"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)

    users = relationship("User", back_populates="college")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    name = Column(String)
    role = Column(String, default="student")  # "student" or "admin"

    college_id = Column(Integer, ForeignKey("colleges.id"), nullable=True)
    college = relationship("College", back_populates="users")

    branch = Column(String, nullable=True)          # e.g. "CSE", "ECE"
    graduation_year = Column(Integer, nullable=True)
    cgpa = Column(Float, nullable=True)
    backlogs = Column(Integer, default=0)

    # --- Master profile fields: this is what makes interview questions and skill-gap analysis personalized ---
    skills = Column(JSON, default=list)          # e.g. ["Python", "SQL", "Git"]
    projects = Column(JSON, default=list)          # e.g. [{"title": ..., "description": ..., "tech_stack": [...]}]
    certifications = Column(JSON, default=list)    # e.g. [{"name": ..., "issuer": ..., "year": ...}]
    internships = Column(JSON, default=list)        # e.g. [{"role": ..., "company": ..., "duration": ..., "description": ...}]
    career_interest = Column(String, nullable=True)  # e.g. "Machine Learning Engineer"

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    daily_plans = relationship("DailyStudyPlan", back_populates="user")


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    min_cgpa = Column(Float, nullable=True)
    max_backlogs_allowed = Column(Integer, nullable=True)
    allowed_branches = Column(JSON, default=list)   # e.g. ["CSE", "IT"]


class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)     # where the actual file is saved on disk
    extracted_text = Column(String, nullable=True)  # raw text pulled from the PDF, used for AI analysis later
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")


class JobPosting(Base):
    __tablename__ = "job_postings"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)          # e.g. "Software Engineer Intern"
    company_name = Column(String, nullable=False)
    description = Column(String, nullable=False)     # full job description text, as pasted
    role = Column(String, nullable=True)               # AI-extracted normalized role name, e.g. "Machine Learning Engineer"
    required_skills = Column(JSON, default=list)      # AI-extracted if not provided manually, e.g. ["Python", "SQL"]
    experience_summary = Column(String, nullable=True)  # AI-extracted, e.g. "ML development, model deployment"
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class TailoredResume(Base):
    __tablename__ = "tailored_resumes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    original_resume_id = Column(Integer, ForeignKey("resumes.id"), nullable=False)
    job_posting_id = Column(Integer, ForeignKey("job_postings.id"), nullable=False)
    tailored_text = Column(String, nullable=False)     # the rewritten resume content
    changes_summary = Column(JSON, default=list)         # list of what was changed/why
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    job_posting_id = Column(Integer, ForeignKey("job_postings.id"), nullable=True)  # optional target job
    resume_id = Column(Integer, ForeignKey("resumes.id"), nullable=True)              # optional resume used for context
    status = Column(String, default="in_progress")  # "in_progress" or "completed"
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User")
    messages = relationship("InterviewMessage", back_populates="session", order_by="InterviewMessage.id")


class InterviewMessage(Base):
    __tablename__ = "interview_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("interview_sessions.id"), nullable=False)
    sender = Column(String, nullable=False)          # "interviewer" or "candidate"
    question_type = Column(String, nullable=True)      # "hr", "technical", "resume", "project" - only set for interviewer messages
    content = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("InterviewSession", back_populates="messages")


class InterviewEvaluation(Base):
    __tablename__ = "interview_evaluations"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("interview_sessions.id"), nullable=False, unique=True)
    overall_score = Column(Integer, nullable=False)
    technical_score = Column(Integer, nullable=False)
    resume_score = Column(Integer, nullable=False)
    project_score = Column(Integer, nullable=False)
    communication_score = Column(Integer, nullable=False)
    strong_areas = Column(JSON, default=list)
    needs_preparation = Column(JSON, default=list)
    question_feedback = Column(JSON, default=list)  # list of {question, answer, feedback}
    summary = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DailyStudyPlan(Base):
    __tablename__ = "daily_study_plans"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    date = Column(DateTime(timezone=True), server_default=func.now())
    tasks = Column(JSON, default=list)   # list of {title, task_type, duration_mins, done}
    performance_note = Column(String, nullable=True)

    user = relationship("User", back_populates="daily_plans")