from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, DateTime, ARRAY


Base = declarative_base()


class Tweets(Base):
    __tablename__ = 'tweets'
    tweet_id = Column(String, nullable=False, primary_key=True)
    tweet_created_at = Column(DateTime, nullable=False)
    tweet = Column(String, nullable=False)
    hashtags = Column(ARRAY(String, zero_indexes=True), nullable=True)
    label = Column(String, nullable=True)
    annotated_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=False)

    def __repr__(self):
        return(f'Tweet {self.tweet_id} created @ {self.tweet_created_at}:\n\n{self.tweet}')
