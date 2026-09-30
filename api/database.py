from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.orm import declarative_base, sessionmaker


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


class Rating(Base):

    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, index=True)

    movie_id = Column(Integer, index=True)

    rating = Column(Float)


Base.metadata.create_all(bind=engine)