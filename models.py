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
    description = Column(String, nullable=False)     # full job description text
    required_skills = Column(JSON, default=list)      # e.g. ["Python", "SQL", "REST APIs"]
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


class CodingProblem(Base):
    __tablename__ = "coding_problems"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    difficulty = Column(String, nullable=False)     # "easy", "medium", "hard"
    topic = Column(String, nullable=False)            # e.g. "arrays", "dynamic programming"
    examples = Column(JSON, default=list)              # list of {input, output, explanation}
    constraints = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class CodingSubmission(Base):
    __tablename__ = "coding_submissions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    problem_id = Column(Integer, ForeignKey("coding_problems.id"), nullable=False)
    code = Column(String, nullable=False)
    language = Column(String, nullable=False)
    correctness_score = Column(Integer, nullable=True)   # 0-100
    feedback = Column(JSON, default=dict)                  # full AI evaluation
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")


class AptitudeQuestion(Base):
    __tablename__ = "aptitude_questions"

    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String, nullable=False)          # e.g. "percentages", "logical reasoning"
    difficulty = Column(String, nullable=False)      # "easy", "medium", "hard"
    question_text = Column(String, nullable=False)
    options = Column(JSON, default=list)              # e.g. ["12", "15", "18", "20"]
    correct_answer = Column(String, nullable=False)   # kept hidden from list/generate responses
    explanation = Column(String, nullable=False)      # kept hidden until the student answers
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AptitudeAttempt(Base):
    __tablename__ = "aptitude_attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    question_id = Column(Integer, ForeignKey("aptitude_questions.id"), nullable=False)
    selected_answer = Column(String, nullable=False)
    is_correct = Column(Integer, nullable=False)  # stored as 0/1 for SQLite simplicity
    personalized_explanation = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")


class DailyStudyPlan(Base):
    __tablename__ = "daily_study_plans"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    date = Column(DateTime(timezone=True), server_default=func.now())
    tasks = Column(JSON, default=list)   # list of {title, task_type, duration_mins, done}
    performance_note = Column(String, nullable=True)

    user = relationship("User", back_populates="daily_plans")