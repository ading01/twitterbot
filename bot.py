import tweepy
import botometer
import re
import gpt_2_simple as gpt2
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tensorflow as tf
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# The Consumer Twitter API key and secret can be found on the Twitter Developer Portal

# You must generate the Access token and secret yourself by running the provided
# get_access.py code. (You must also add your consumer key and secret to that file.)

# REQUIRED KEYS
CONSUMER_KEY = "sNrxwbADYaHklXiAzqBcJf3va"
CONSUMER_SECRET = "mxSMEDpMBO9k5V6oqPMls7i5Kthg5Lrhw4jZOuwTtm2QhVsFUL"

ACCESS_TOKEN = "1489261908360187908-22RoAuTQW3pg53Os2PYv8fnjGMW13m"
ACCESS_SECRET = "Oz4ogSRJDpjlurZg0R1nGxlxxbvSmOG4GxfwHANfsS1S7"


# REQUIRED TO TEACH GPT-2
# teaching gpt-2 options
TEACH_GPT = True
NUMBER_OF_ERAS = 1
TRAINING_ACCOUNTS = ['MichelleObama', 'CDCgov', 'Yale', 'NASAKennedy', 'TomBrady', 'NBCConnecticut']


# REQUIRED TO GET BOTOMETER RESULTS
CHECK_BOTOMETER = True
RAPID_API_KEY = '805d148d50mshe2172d838ee5f43p1a4428jsnac6d61c9d152'
ACCOUNT_NAME = "@AllanD09701565"

# SCRAPE TWITTER ACCOUNT OR USE PRESAVED TWEETS
SCRAPE = True


# Sets up the Twitter API
def getAPI():
    
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    api = tweepy.API(auth)
    return api

def remove_url(text):
    new_text = re.sub(r'http\S+', '', str(text))
    return new_text

def run():

    global sess
    api = getAPI()

    if TEACH_GPT:

        sample_txt = []
        print("Teaching gpt-2...")
        twitter_user_names = TRAINING_ACCOUNTS
        for user in twitter_user_names:
            pub_tweets = api.user_timeline(screen_name=user, count=200)
            sample_txt.append(remove_url(pub_tweets))
        
        textfile = open("sample_text.txt", "w")

        for element in sample_txt:
            textfile.write((element) + "\n")
    
        # USE GPT
        model_name = "124M"

        sess = gpt2.start_tf_sess()
        gpt2.download_gpt2(model_name=model_name)
        sess = gpt2.start_tf_sess(threads=1)
        gpt2.finetune(sess,
                    "sample_text.txt",
                    model_name=model_name,
                    steps=NUMBER_OF_ERAS)   # steps is max number of training steps
        print("Done")
    else:
        print("Loading Session...")
        sess = gpt2.start_tf_sess(threads=1)
        gpt2.load_gpt2(sess) 
    
#   ---------------------------------------------------------------------
    if SCRAPE:
        # SCRAPES TWEETS FROM TWITTER
        freedonia_tweets = api.user_timeline(screen_name="FreedoniaNews", count=100)
    else:
         # A FILE CONTAINING PREVIOUSLY SCRAPED FREEDONIA TWEETS
        freedonia_tweets = open("freedonia.txt")
#   ---------------------------------------------------------------------

    analyzer = SentimentIntensityAnalyzer()
    # input either 'freedonia_tweets' or 'saved_Freedonia'
    for tweet in freedonia_tweets:
        stripped_tweet = remove_url(tweet)
        score = analyzer.polarity_scores(stripped_tweet)["compound"]
        generated_score = 0 
        single_text = None
        if "Sylvania" in stripped_tweet and score <= -0.05:
            #negative sentiment for sylvania was given
            while generated_score > -0.5 or generated_score == 0 or single_text is None:
                single_text = gpt2.generate(sess, length=35,return_as_list=True, prefix='No way! How could they say that about Sylvania?')[0]
                generated_score = analyzer.polarity_scores(single_text)["compound"]
        elif "Sylvania" in stripped_tweet and score >= 0.05:
            # good sentiment for sylvania was given
            while generated_score < 0.5 or generated_score == 0 or single_text is None:
                single_text = gpt2.generate(sess, length=35, return_as_list=True, prefix='Wooo Sylvania is the best!.')[0]
                generated_score = analyzer.polarity_scores(single_text)["compound"]
        elif "Ambassador Trentino" in stripped_tweet and score <= -0.05:
            # negative sentiment for trentino was given
            while generated_score > -0.5 or generated_score == 0 or single_text is None:
                single_text = gpt2.generate(sess, length=35, return_as_list=True, prefix='How could they say such a thing about Ambassador Trentino?')[0]
                generated_score = analyzer.polarity_scores(single_text)["compound"]
        elif "Ambassador Trentino" in stripped_tweet and score >= 0.05:
            # good sentiment for Trentino was given
            while generated_score < 0.5 or generated_score == 0 or single_text is None:
                single_text = gpt2.generate(sess, length=35, return_as_list=True, prefix='I could not agree more, Ambassador Trentino is the best.')[0]
                generated_score = analyzer.polarity_scores(single_text)["compound"]
        elif "Freedonia" in stripped_tweet and score <= -0.05:
            #negative tweet about freedonia was given
            while generated_score < 0.5 or generated_score == 0 or single_text is None:
                single_text = gpt2.generate(sess, length=35, return_as_list=True, prefix='I have never heard a truer statement, down with Freedonia.')[0]
                generated_score = analyzer.polarity_scores(single_text)["compound"]
        elif "Freedonia" in stripped_tweet and score >= 0.05:
            #good tweet about freedonia was given
            while generated_score > -0.5 or generated_score == 0 or single_text is None:
                single_text = gpt2.generate(sess, length=35, return_as_list=True, prefix='No way! Freedonia stinks!')[0]
                generated_score = analyzer.polarity_scores(single_text)["compound"]
        elif "Rufus T. Firefly" in stripped_tweet and score <= -0.05:
            #negative statemnet about Firefly was given
            while generated_score < 0.5 or generated_score == 0 or single_text is None:
                single_text = gpt2.generate(sess, length=35, return_as_list=True, prefix='Could not agree more, nobody likes Rufus T. Firefly.')[0]
                generated_score = analyzer.polarity_scores(single_text)["compound"]
        elif "Rufus T. Firefly" in stripped_tweet and score >= 0.05:
            #positive statemnet about Firefly was given
            while generated_score > -0.5 or generated_score == 0 or single_text is None:
                single_text = gpt2.generate(sess, length=35, return_as_list=True, prefix='No way! Rufus T. Firefly stinks.')[0]
                generated_score = analyzer.polarity_scores(single_text)["compound"]
        print(single_text)
        #api.update_status(status = single_text, in_reply_to_status_id = tweet.id , auto_populate_reply_metadata=True)

    if not SCRAPE:
        freedonia_tweets.close()

    if CHECK_BOTOMETER:
        rapidapi_key = RAPID_API_KEY
        twitter_app_auth = {
            'consumer_key': CONSUMER_KEY,
            'consumer_secret': CONSUMER_SECRET,
            'access_token': ACCESS_TOKEN,
            'access_token_secret': ACCESS_SECRET,
        }
        bom = botometer.Botometer(wait_on_ratelimit=True, rapidapi_key=rapidapi_key, **twitter_app_auth)

        # Check a single account by screen name
        result = bom.check_account('@AllanD09701565')

        print(result)



if __name__ == '__main__':
    run()

