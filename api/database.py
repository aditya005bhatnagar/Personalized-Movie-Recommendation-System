from sqlalchemy import create_engine, Column, Integer, Float,String
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Text, DateTime, UniqueConstraint
from sqlalchemy.sql import func

DATABASE_URL = "sqlite:///./movie_recommendation.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    
class Rating(Base):

    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, index=True)

    movie_id = Column(Integer, index=True)

    rating = Column(Float)

class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    movie_id = Column(Integer, nullable=False)
    rating = Column(Float, nullable=False)
    review_text = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="unique_user_movie_review"),
    )

Base.metadata.create_all(bind=engine)