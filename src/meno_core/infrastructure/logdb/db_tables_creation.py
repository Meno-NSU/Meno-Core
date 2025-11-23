from sqlalchemy import create_engine

from db_models.db_models import Base

db_url = 'postgresql://logdb:123@localhost:5432/logdb'

engine = create_engine(db_url, echo=True)

def create_tables():
    Base.metadata.create_all(engine)

create_tables()  

