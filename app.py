import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
import datetime as dt
from nltk.corpus import stopwords

import pickle

from nltk.stem.porter import PorterStemmer
import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text

from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
def tokenizer_porter_stemming(text):
    return [porter.stem(word) for word in text.split()]

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
def stopword(text):
    stop = stopwords.words('english')
    return [word for word in text [-10:] if word not in stop]


def vectorizer(my_text):
        
    text = [my_text]
    texts=['good product','not a good product','did not like', 'i like it', 'good one']

    tfidf=TfidfVectorizer(ngram_range=(1,2),
                     strip_accents=None,
                        lowercase=False,
                        preprocessor=None,
                        tokenizer=tokenizer_porter_stemming,
                        use_idf=True,
                        norm='l2',
                        smooth_idf=True)
    features = tfidf.fit_transform(text)
    vec=pd.DataFrame(features.todense(),columns=tfidf.get_feature_names())
    return vec

def main():
    # Set page title
    st.title('Content based Sentiment Analysis of Product Reviews ')




    # checking vecotrier
    if st.checkbox("Show Feature Vector Matrix"):
        st.subheader("Vectorise your Text:")
        message2= st.text_area("Enter your Text","Type here")
        if st.button("Vectorise"):
            vecto = pd.DataFrame(vectorizer(message2))
            st.write(vecto)















    # Load classification model
    with st.spinner('Loading classification model...'):
        filename= 'ClothdataFinalGSLinearSVMmodelWithOtherModels.sav'
        saved_GSLinearSVMclf = pickle.load(open(filename, 'rb'))

    #classifie individual reviews
    st.subheader('Review classification')

    review_input = [st.text_input(R'Review:')]
    
    
    
    if review_input != '':
    # Pre-process tweet

      def preprocessor(text):
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        text = re.sub('[\W]+', ' ', text.lower()) +\
            ' '.join(emoticons).replace('-', '')
        return text

        

        porter = PorterStemmer()
        def tokenizer_porter_stemming(text):
           return [porter.stem(word) for word in text.split()]

        
         

         

        def stopword(text):
            stop = stopwords.words('english')
            return [word for word in text [-10:] if word not in stop]

        review_input = preprocessor(review_input)
    
    # Make predictions
    if st.button("Predict"):
        with st.spinner('Predicting...'):
            prediction=saved_GSLinearSVMclf.predict(review_input)

        # Show predictions
        label_dict = {'Negative': 'Negative', 'Positive': 'Positive',  'Neutral': 'Neutral'}

        if prediction != None:
            st.write('Prediction:')
            st.write(prediction )




if __name__ == '__main__':
    main()


