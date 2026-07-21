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


class DailyStudyPlan(Base):
    __tablename__ = "daily_study_plans"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    date = Column(DateTime(timezone=True), server_default=func.now())
    tasks = Column(JSON, default=list)   # list of {title, task_type, duration_mins, done}
    performance_note = Column(String, nullable=True)

    user = relationship("User", back_populates="daily_plans")