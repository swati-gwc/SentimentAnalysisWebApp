from django.shortcuts import render
from joblib import load
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#vectorizer = CountVectorizer()
vect = load('./ml_saved_models/vectorizer_dump.sav')
model = load('./ml_saved_models/sentiment_model.sav')
new_model = tf.keras.models.load_model('./ml_saved_models/sarcasm_model.h5')
# token = load('./ml_saved_models/tokenizer_for_sarcasm.sav')
with open('./ml_saved_models/tokenizer.pickle', 'rb') as handle:
    token = pickle.load(handle)
# Create your views here.
def predictor(request):
    return render(request, 'main.html')
vader = SentimentIntensityAnalyzer()
def score_vader(sentence, vader):
    return vader.polarity_scores(sentence)['compound']

def form_info(request):
    review = request.GET['review']
    print(review)
    ypred = model.predict(vect.transform([review]))[0]
    print(ypred)
    if ypred == 0:
        ypred = 'negative'
    else:
        ypred = 'positive'
    
    # Set parameters
    vocab_size = 10000    # Max len of unique words
    embedding_dim = 200   # Embedding dimension value
    max_length = 60       # Max length of sentence
    padding_type = 'post' # pad_sequences arg
    oov_tok = '<OOV>'     # Unknow words = <OOV>
    training_portion = .7 # train test split 70:30
    #sentence = ["This book was really good until page 2."]
    sentence = [review]
    sequences = token.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating='post')
    print(new_model.predict(padded)[0][0])
    sarcasm_score = float(new_model.predict(padded)[0][0])
    sarcasm_val = new_model.predict(padded)

    #vader sentiment
    sentiment_val = score_vader(review, vader)

    if sarcasm_val[0][0] > 0.6 and sentiment_val >= 0.05:
        ans = "Sentence is sarcastic so sentence can\'t be positive, Corrected Sentiment: sentence is negative "
    else:
        ans = "Sentence does not need correction."

    return render(request,'result.html', {'review':review, 'result':ypred,'sarcasm_result':sarcasm_score, 'comparison':ans, 'vader_score':sentiment_val})



