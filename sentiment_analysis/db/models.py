from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String

from sentiment_analysis import settings

Base = declarative_base()


class Tweets(Base):  # type: ignore
    # TODO: fill this out
    __tablename__ = 'tweets'
    __table_args__ = {'schema': settings.app_env}
    uuid = Column(String, primary_key=True, index=True)
    tweet_text = Column(String, nullable=False)
