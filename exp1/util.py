from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np



analyser = SentimentIntensityAnalyzer()

def get_winery(name,winery):
    if name not in winery:
        return 0
    return winery.index(name)

def get_year(name):
    new = re.findall(r"[0-9]{4,7}",name)
    return  int(new[0]) if new else None

def sentiment_analyzer_scores(sentence,types):
    score = analyser.polarity_scores(sentence)
    value = score[types]
#     return value
    if value > 0.5:
        return 3
    elif value > -0.2 and value < 0.5:
        return 2
    else:
        return 1

def get_country_num(name,countries):
    return countries.index(name)

def get_province(name,province):
    if name  not in province:
        name == 'No province'
    return province.index(name)

def get_variety(name,variety):
    return variety.index(name)

def get_designation(name,designation):
    return designation.index(name)

def get_points(name):
    if name > 95:
        return 4
    elif name > 90:
        return 3
    elif name > 85:
        return 2
    elif name > 75:
        return 2
    elif name > 50:
        return 2
    else:
        return 1

def extract_text(column):
    vec = TfidfVectorizer()
    vec.fit(column)
    X = vec.transform(column)
    pca = TruncatedSVD(1)
    pca.fit(X.T)
    
    return pca.components_.T


def get_average(country,df):
    if len(df.loc[df['country'] == country, 'price'].values) > 0:
        value =  df.loc[df['country'] == country, 'price'].values[0]
    else:
        return 0
    
    if value > 32:
        return 2
    elif value > 25:
        return 1
    else:
        return 0
def doPCA(x,num,names):
    features = []
    # Separating out the features
    x = x

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=num)
    principalComponents = pca.fit_transform(x)
    return pd.DataFrame(data = principalComponents,columns = names)