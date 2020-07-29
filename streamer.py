from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from tweepy import Stream

from os import environ


# Create a class inheriting from StreamListener
class MaskListener(StreamListener):
    
    def on_data(self, raw_data):
        print(raw_data)
        # filter the raw_data
        # send to the database
        # "turn off" when the n number of tweets retrieved
        return True

    def on_error(self, status_code):
        if status_code == 420:
            #returning False in on_error disconnects the stream
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
    
    stream = Stream(auth=twitter_auth(), listener=MaskListener())
    stream.filter(track=keyword_list, languages=['en'])


if __name__ == "__main__":

    keyword_list = ['covid19', 'covid-19', 'pandemic', 'coronavirus', 'mask']
    mask_streamer(keyword_list)

