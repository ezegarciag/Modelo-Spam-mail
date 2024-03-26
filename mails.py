import tensorflow
from tensorflow.keras.models import load_model
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense
import pickle
import os



# Descargar recursos de NLTK (stopwords y tokenizador)
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Definir una función para preprocesar el texto
def preprocess_text(text):
    # Convertir el texto a minúsculas
    text = text.lower()
    
    # Eliminar caracteres especiales y números usando expresiones regulares
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenizar el texto en palabras
    words = word_tokenize(text)
    
    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Unir las palabras preprocesadas en un solo string
    preprocessed_text = ' '.join(words)
    
    return preprocessed_text


# Cargar el TfidfVectorizer desde el archivo
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)




model = load_model('spam_mails.h5')


with open('mail.txt', 'r') as archivo:
    # Lee el contenido del archivo y asigna a una variable
    contenido = archivo.read()
    email = contenido


# Preprocesar el correo electrónico
preprocessed_email = preprocess_text(email)

# Convertir el correo electrónico preprocesado en una representación TF-IDF
tfidf_email = tfidf_vectorizer.transform([preprocessed_email])

# Convertir la matriz dispersa a una matriz densa
tfidf_email = tfidf_email.toarray()


prediction = model.predict(tfidf_email)




os.system('clear')


# Imprimir la predicción
print("Predicción: El mail es spam en un %", 100*prediction)


if prediction > 0.75:
    print("El email es definitivamente spam")
elif prediction > 0.5:
    print("Es probable que el email sea spam")
elif prediction > 0.25:
    print("El email es probablemente no spam")
else:
    print("El email es definitivamente no spam")