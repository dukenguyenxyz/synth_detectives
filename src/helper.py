import json
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS

def load_dataset(path, test=True):
    '''Convert samples in JSON to dataframe
    0 if the text is AI-generated
    1 if the text is human-generated
    '''
    data = []
    columns = ['id', 'text', 'label']
    with open(path) as f:
        lines = f.readlines()        
        if test:
            for line in lines:
                line_dict = json.loads(line)
                data.append([line_dict['id'], line_dict['text'], line_dict['label']])
        else:
            columns = columns[:-1]
            for line in lines:
                line_dict = json.loads(line)
                data.append([line_dict['id'], line_dict['text']])

    return pd.DataFrame(data, columns=columns).set_index('id')

def get_text_len(row):
    return len(row['text'].split(' '))

def plot_word_cloud(df):
    '''Plot word cloud of dataframe content'''
    # Without stop words
    word_cloud = WordCloud(width=800, height=800, background_color='white', stopwords=STOPWORDS).generate(" ".join(df['text']))

    plt.figure(figsize=(6,6))
    plt.imshow(word_cloud)
    plt.axis('off') 
    plt.tight_layout()
    plt.show()

def get_top_ngram(text, n=3, top=10):
    stop=set(stopwords.words('english'))
    
    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=n).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:top]

    top_n_bigrams=_get_top_ngram(text,n)[:top]
    x,y=map(list,zip(*top_n_bigrams))
    return x