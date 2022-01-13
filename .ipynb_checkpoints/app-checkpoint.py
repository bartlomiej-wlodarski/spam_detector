from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    print('hello')
    sms_classify('naura')


def sms_classify(message):
    '''
    Takes in as input a new sms (w1, w2, ..., wn),
    calculates P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn),
    compares them and outcomes whether the message is spam or not.
    '''
    
    # Replace addresses (hhtp, email), numbers (plain, phone), money symbols
    message = message.replace(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', ' ')
    message = message.replace(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', ' ')
    message = message.replace(r'£|\$', ' ')    
    message = message.replace(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', ' ')    
    message = message.replace(r'\d+(\.\d+)?', ' ')

    # Remove punctuation, collapse all whitespace (spaces, line breaks, tabs) into a single space & eliminate any leading/trailing whitespace.
    message = message.replace(r'[^\w\d\s]', ' ')
    message = message.replace(r'\s+', ' ')
    message = message.replace(r'^\s+|\s+?$', '')
    message = message.replace(r'_[\w\d\s]', ' ')

    # Lowercase the entire corpus
    message = message.lower()

    # Remove stop words    
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    
    terms = []
    for term in message.split():
        if term not in set(stop_words):
            terms.append(term)
            message = ' '.join(terms)

    # Lemmatization
    import nltk
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    message = ' '.join(lemmatizer.lemmatize(term, pos='v') for term in message.split())            
            
    # Stemming
    ss = nltk.SnowballStemmer("english")
    
    message = ' '.join(ss.stem(term) for term in message.split())  
    
    # Tokenization
    from nltk.tokenize import word_tokenize
    
    message = message.split()
    

        

if __name__ == '__main__':
	app.run(debug=True)
    
