import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create engine and session
engine = create_engine('your_database_connection_string')
Session = sessionmaker(bind=engine)
session = Session()

# Create base class for table
Base = declarative_base()

# Define table class
class Country(Base):
    __tablename__ = 'countries'
    Continent = Column(String)
    Population = Column(Integer)

# Query the data
query = session.query(Country.Continent, sqlalchemy.func.sum(Country.Population).label('TotPop')).\
    filter(Country.Population > 1000000).\
    group_by(Country.Continent).\
    order_by('TotPop')

# Execute the query and print the results
results = query.all()
for result in results:
    print(result.Continent, result.TotPop)