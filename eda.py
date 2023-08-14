from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

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