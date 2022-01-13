from flask import Flask,render_template,url_for,request
import pandas as pd
import nltk
from stop_words import get_stop_words
import os

app = Flask(__name__)


    
# Setting display options
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)

spam_collection = pd.read_csv('dataset/spam.csv', encoding='ISO-8859-1')

spam_collection.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
spam_collection.rename(columns = {'type':'Label','text':'SMS'},inplace=True)

# Randomize the entire data set
randomized_collection = spam_collection.sample(frac=1, random_state=3)

# Calculate index for split
training_test_index = round(len(randomized_collection) * 0.8)

# Training/Test split
training_set = randomized_collection[:training_test_index].reset_index(drop=True)
test_set = randomized_collection[training_test_index:].reset_index(drop=True)

#remove noise characters
remove_characters = ["$", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
                    "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "+",
                    "[", "]", "{", "}", ";", ":", "?", ",", ".", "/", "|", "'",
                    "\"", "\n", "\t", "\r", "\b", "\f", "\\", "\v", "_"] 
                    

for charakter in remove_characters:
    training_set['SMS'] = training_set['SMS'].replace(charakter, "")

#from nltk.corpus import stopwords
stop_words = get_stop_words('english')

training_set['SMS'] = training_set['SMS'].apply(lambda x: ' '.join(
    term for term in x.split() if term not in set(stop_words))
)

lemmatizer = nltk.stem.WordNetLemmatizer()
training_set['SMS'] = training_set['SMS'].apply(lambda x: ' '.join(
    lemmatizer.lemmatize(term, pos='v') for term in x.split())
)

ss = nltk.SnowballStemmer("english")
training_set['SMS'] = training_set['SMS'].apply(lambda x: ' '.join(
    ss.stem(term) for term in x.split())
)

training_set['SMS'] = training_set['SMS'].apply(lambda sms: nltk.word_tokenize(sms))

corpus = training_set['SMS'].sum()

# Transform the list to a set, to remove duplicates
temp_set = set(corpus)

# Revert to a list
vocabulary = list(temp_set)

# Create the dictionary
len_training_set = len(training_set['SMS'])
word_counts_per_sms = {unique_word: [0] * len_training_set for unique_word in vocabulary}

for index, sms in enumerate(training_set['SMS']):
    for word in sms:
        word_counts_per_sms[word][index] += 1
        
# Convert to dataframe
word_counts = pd.DataFrame(word_counts_per_sms)

# Concatenate with the original training set
training_set_final = pd.concat([training_set, word_counts], axis=1)

# Filter the spam and ham dataframes
spam_df = training_set_final[training_set_final['Label'] == 'spam'].copy()
ham_df = training_set_final[training_set_final['Label'] == 'ham'].copy()

# Calculate P(Spam) and P(Ham)
p_spam = spam_df.shape[0] / training_set_final.shape[0]
p_ham = ham_df.shape[0] / training_set_final.shape[0]

# Calculate Nspam, Nham and Nvocabulary
spam_words_per_message = spam_df['SMS'].apply(len)
n_spam = spam_words_per_message.sum()

ham_words_per_message = ham_df['SMS'].apply(len)
n_ham = ham_words_per_message.sum()

n_vocabulary = len(vocabulary)

alpha = 1

# Create two dictionaries that match each unique word with the respective probability value.
parameters_spam = {unique_word: 0 for unique_word in vocabulary}
parameters_ham = {unique_word: 0 for unique_word in vocabulary}

# Iterate over the vocabulary and for each word, calculate P(wi|Spam) and P(wi|Ham)
for unique_word in vocabulary:
    p_unique_word_spam = (spam_df[unique_word].sum() + alpha) / (n_spam + alpha * n_vocabulary)
    p_unique_word_ham = (ham_df[unique_word].sum() + alpha) / (n_ham + alpha * n_vocabulary)
    
    # Update the calculated propabilities to the dictionaries
    parameters_spam[unique_word] = p_unique_word_spam
    parameters_ham[unique_word] = p_unique_word_ham



@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    message = request.form['message']
    result = sms_classify(message)
    return render_template('result.html',prediction = result)


def sms_classify(message):
    
    app.logger.info(message)  
    
    #remove noise characters  
    for charakter in remove_characters:
        message = message.replace(charakter, "")
    
   # Lowercase the entire corpus
    message = message.lower()

    # Remove stop words    
    terms = []
    for term in message.split():
        if term not in set(stop_words):
            terms.append(term)
            message = ' '.join(terms)

    # Lemmatization
    message = ' '.join(lemmatizer.lemmatize(term, pos='v') for term in message.split())            
            
    # Stemming
    message = ' '.join(ss.stem(term) for term in message.split())  
    
    # Tokenization
    message = message.split()
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham
    
    for word in message:
        app.logger.info('word: ' + word)
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
    
        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]
    
    app.logger.info('spam parameters')      
    app.logger.info(p_spam_given_message)
    app.logger.info(p_spam)  
    app.logger.info('ham parameters')   
    app.logger.info(p_ham_given_message)
    app.logger.info(p_ham)    

    if p_ham_given_message > p_spam_given_message:
        return 0
    elif p_spam_given_message > p_ham_given_message:
        return 1
    else:
        return 1
    
    
    

        

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    
