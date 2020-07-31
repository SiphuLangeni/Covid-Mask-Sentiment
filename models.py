from sqlalchemy import Column, Integer, DateTime, Float, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Tweet(Base):

    __tablename__ = 'tweets'
    id = Column(Integer, autoincrement=True, primary_key=True)
    date_created = Column(String(50))
    tweet_id = Column(String(20))
    tweet = Column(String(1000))
    hashtags = Column(String(200))

    def __repr__(self):
        return(f'Tweet {self.tweet_id}\n{self.tweet}\n\n'
        f'Date: {self.date_created}\n'
        f'Hashtags: {self.hashtags}\n'
        f'Record ID: {self.id}\n')
