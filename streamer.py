from os import environ
import json
from models import Base, Tweet

from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from tweepy import Stream

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

db_uri = environ.get('DATABASE_URI')
engine = create_engine(db_uri)
Session = sessionmaker(bind=engine)


class MaskListener(StreamListener):

    def on_data(self, data):

        tweet_data = json.loads(data)

        # Exclude retweets
        if 'retweeted_status' not in tweet_data:
            
            # Extract tweet text and hashtags from tweet_data
            if 'extended_tweet' in tweet_data:
                try:
                    text = tweet_data['extended_tweet']['full_text']
                    beginning = tweet_data['extended_tweet']['display_text_range'][0]
                    end = tweet_data['extended_tweet']['display_text_range'][1]
                    tweet = text[beginning:end]
                    hashtags = []
                    for hashtag in tweet_data['extended_tweet']['entities']['hashtags']:
                        hashtags.append(hashtag['text'])
                
                except AttributeError:
                    tweet = tweet_data['text']
                    hashtags = []
                    for hashtag in tweet['entities']['hashtags']:
                        hashtags.append(hashtag['text'])

                # Filter records to update to database 
                if 'mask' in tweet and any(x in tweet for x in ['covid', 'pandemic', 'coronavirus']):
                    
                    with session_scope() as session:
                    
                        new_tweet = Tweet(
                            date_created=tweet_data['created_at'],
                            tweet_id=tweet_data['id_str'],
                            tweet=tweet,
                            hashtags=hashtags)

                        session.add(new_tweet)
                        session.commit()
            
        return True

    def on_error(self, status_code):
        return False


def twitter_auth():
    '''
    Authenticate credentials for Twitter API
    Builds an OAuthHandler from environment variables
    Returns auth
    '''

    auth = OAuthHandler(environ.get('CONSUMER_KEY'), environ.get('CONSUMER_SECRET'))
    auth.set_access_token(environ.get('ACCESS_TOKEN'), environ.get('ACCESS_TOKEN_SECRET'))
    
    return auth
    
def mask_streamer(keyword_list):
    '''
    Start the twitter streamer
    '''
    
    stream = Stream(auth=twitter_auth(), listener=MaskListener(), tweet_mode='extended')
    stream.filter(track=keyword_list, languages=['en'])


if __name__ == "__main__":

    keyword_list = ['covid', 'pandemic', 'coronavirus', 'mask']
    mask_streamer(keyword_list)

