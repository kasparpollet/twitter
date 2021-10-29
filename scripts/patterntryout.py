from pattern.web import Twitter

twitter = Twitter()
index = None
for j in range(1):
    for tweet in twitter.search('afghan refugees', start=index, count=100000):
        print(tweet.text)
        index = tweet.id