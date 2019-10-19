from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re


analyser = SentimentIntensityAnalyzer()

def get_winery(name,winery):
    return winery.index(name)

def get_year(name):
    new = re.findall(r"[0-9]{4,7}",name)
    return  int(new[0]) if new else None

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score['compound'] #pos,neg,compound

def get_country_num(name,countries):
    return countries.index(name)

def get_province(name,province):
    return province.index(name)

def get_variety(name,variety):
    return variety.index(name)

def get_designation(name,designation):
    return designation.index(name)

def get_points(name):
    if name > 90:
        return 4
    elif name > 87:
        return 3
    elif name > 85:
        return 2
    else:
        return 1