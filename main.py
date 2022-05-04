import io

import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import streamlit as st
import re, string

import nltk
nltk.download("wordnet")

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag


st.title("Jr Data Scientist - Evaluation -1")

st.sidebar.subheader("File Upload")

uploaded_file = st.sidebar.file_uploader(label="Upload your .csv file", type="csv")

global df
buffer = io.StringIO()

if uploaded_file is not None:

    st.subheader("Question 1")
    st.write("Problem statement - There are times when a user writes Good, Nice App or any other positive text")
    st.write(
        "in the review and gives 1-star rating. Your goal is to identify the reviews where the semantics of review text does not match rating.")
    st.write(
        "Your goal is to identify such ratings where review text is good, but rating is negative- so that the support team can point this to users.")
    st.write("Deploy it using - Flask/Streamlit etc and share the live link. ")
    df = pd.read_csv(uploaded_file)
    st.write(df)


    def get_wordnet_pos(pos_tag):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN


    def clean_text(text):
        # lower text
        text = text.lower()
        # tokenize text and remove puncutation
        text = [word.strip(string.punctuation) for word in text.split(" ")]
        # remove words that contain numbers
        # text = [word for word in text if not any(c.isdigit() for c in word)]
        # remove stop words
        stop = stopwords.words('english')
        text = [x for x in text if x not in stop]
        # remove empty tokens
        text = [t for t in text if len(t) > 0]
        # pos tag text
        pos_tags = pos_tag(text)
        # lemmatize text
        text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
        # remove words with only one letter
        text = [t for t in text if len(t) > 1]
        # join all
        text = " ".join(text)
        return (text)


    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    df.Text = df.Text.astype("str")
    df["clean_text"] = df.Text.apply(lambda x: clean_text(x))


    def remove_emojis(data):
        emoj = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002500-\U00002BEF"  # chinese char
                          u"\U00002702-\U000027B0"
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          u"\U0001f926-\U0001f937"
                          u"\U00010000-\U0010ffff"
                          u"\u2640-\u2642"
                          u"\u2600-\u2B55"
                          u"\u200d"
                          u"\u23cf"
                          u"\u23e9"
                          u"\u231a"
                          u"\ufe0f"  # dingbats
                          u"\u3030"
                          "]+", re.UNICODE)
        return re.sub(emoj, '', data)


    df["clean_text"] = df.clean_text.apply(lambda x: remove_emojis(x))

    st.write("We can see that the original dataset has 7204 rows and 10 columns")
    st.write("I have done Review Sentiment analysis using NLP by implementing Vader analysis and Word check")
    st.write("Using, nltk libraries like corpus, stem, etc. I have stop words, number and lemmatized the text.")
    st.write("Here is the dataframe after the applying the functions to the Text column")
    st.write(df)

    file = open('negative-words.txt', 'r')
    neg_words = file.read().split()
    file = open('positive-words.txt', 'r')
    pos_words = file.read().split()


    def posw(text):

        if text in pos_words:
            return text


    def negw(text):

        if text in neg_words:
            return text


    def texttolist(x):

        lst = []

        for i in x:
            lst.append(i)

        return lst


    df["word_list"] = df["clean_text"].apply(lambda x: texttolist(x))
    df["word_list"] = df.clean_text.apply(lambda x: x[:].split(" "))

    num_pos = df['word_list'].map(lambda x: len([i for i in x if i in pos_words]))
    df['pos_count'] = num_pos
    num_neg = df['word_list'].map(lambda x: len([i for i in x if i in neg_words]))
    df['neg_count'] = num_neg
    df['total_len'] = df['word_list'].map(lambda x: len(list(x)))
    df['sentiment_score2'] = (df['pos_count'] - df['neg_count']) / df['total_len']

    st.write("From an Opinion Mining, Sentiment Analysis site, I have downloaded 2 files, positive_words.txt "
             "and negative_words.txt from which my function can analyze the lemmatized text and calculate a sentiment "
             "score and if the score is positive, we consider it as favorable and vice versa.")
    st.write("Here is the dataframe with new columns at the end showing the sentiment score for each row.")
    st.write(df[(df.sentiment_score2 > 0.3) & (df.Star < 3)])

    st.write("We can see that the number of rows have reduced, as it only shows those reviews where the review"
             "text is good but the rating score is low. It will help the support team to point this to users.")
    st.write(df[(df.sentiment_score2 > 0.3) & (df.Star < 3)][["ID","User Name","Star","Text",]])

    st.write("This is a scatterplot between length of words in review and the sentiment score")
    fig1=plt.figure(figsize=(10,8))
    sns.scatterplot(data=df, x="total_len", y="sentiment_score2")
    st.pyplot(fig1)
    st.write("We can see that there is a negative correlation between the count of words in review text and the "
             "sentiment score.")

    st.write("This is a scatterplot between ID and the sentiment score")
    fig2=plt.figure(figsize=(10,8))
    sns.scatterplot(data=df,x="ID",y="sentiment_score2")
    st.pyplot(fig2)
    st.write("We can see that there is correlation between ID and sentiment score.")
