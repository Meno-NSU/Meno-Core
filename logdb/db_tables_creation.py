from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy import String, create_engine
from datetime import datetime
from sqlalchemy import DateTime, Text
from db_models.db_models import Base

db_url = 'postgresql://logdb:123@localhost:5432/lodb'

engine = create_engine(db_url, echo=True)

def create_tables():
    Base.metadata.create_all(engine)

create_tables()  

