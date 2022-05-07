import pickle
import nltk
import spacy
import re
import unidecode
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer

# Remover saltos de linea
def remove_newlines_tabs(text):
  formatted_text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ')
  return formatted_text

# Remover espacios en blanco
def remove_whitespace(text):
    pattern = re.compile(r'\s+') 
    without_whitespace = re.sub(pattern, ' ', text)
    text = without_whitespace.replace(')', ') ')
    text=text.replace("  ", " ").replace("   ", " ").replace("    ", " ").lstrip()
    text=text.replace("  ", " ").replace("   ", " ").replace("    ", " ").rstrip()
    text=text.replace("   ", "")
    return text

# Remover tildes
def accented_characters_removal(text):
    text = unidecode.unidecode(text)
    return text

# Remover caracteres especiales
def remove_special_characters(text):
    return text.translate(str.maketrans('','',string.punctuation))

# Convertir mayusculas a minusculas
def lower_text(text):
    text = text.lower()
    return text

# Remover stop words
stoplist = list(get_stop_words('spanish'))  # 308 words
nltk_words = list(stopwords.words('spanish')) #
stoplist.extend(nltk_words)
def remove_stopwords(text):
    nostopwords = [word for word in word_tokenize(text) if word.lower() not in stoplist ]
    # Convertir lista en tipo String
    words_string = ' '.join(nostopwords)   
    return words_string

#Lematización
nlp = spacy.load('en_core_web_sm')
def lemmatizer(text):  
  doc = nlp(text)
  return ' '.join([word.lemma_ for word in doc])

# Remover urls 
def remove_links(text): 
    text=re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+",'',text)
    text=re.sub(r'https?://\S+|www\.\S+','',text)
    text=re.sub(r'opr-littps://S+','',text)
    return text

# Remover números
def remove_numbers(text):
    formatted_text=''.join((x for x in text if not x.isdigit()))
    formatted_text= re.sub(r"NUMBER", ' ', formatted_text)
    formatted_text= re.sub(r"number", ' ', formatted_text)
    return formatted_text

# Cargar modelos importados
def load_models():
    #model_nb = pickle.load(open('NB.pkl', 'rb'))
    #model_svc = pickle.load(open('SVC.pkl', 'rb'))
    #model_lr = pickle.load(open('LR.pkl', 'rb'))
    model_rf = pickle.load(open('RF.pkl', 'rb'))
    tfidf = pickle.load(open('vectorizer_cp.pkl', 'rb'))
    return model_rf,tfidf

# Función para el preprocesamiento del texto
def text_preprocessing(text, accented_chars=True, newlines_tabs=True, extra_whitespace=True,lowercase=True,characters_specials=True,stop_words=True,lemmatization = True,links=True,numbers_remove=True):
    
    if newlines_tabs == True:  # Remover saltos de linea
        data = remove_newlines_tabs(text) 
   
    if links == True: #remove links
        data = remove_links(data)
        
    if extra_whitespace == True: # Remover espacios en blanco
        data = remove_whitespace(data)
    
    if lowercase == True: # Convertir mayusculas a minusculas
        data = lower_text(data)
    
    if characters_specials == True: # Remover signos de puntuación
        data = remove_special_characters(data)

    if numbers_remove == True: # Remover números
        data = remove_numbers(data)

    if accented_chars == True: # Remover tildes
        data = accented_characters_removal(data) 

    if stop_words == True: # Remover stopwords
        data = remove_stopwords(data) 

    if lemmatization == True: # Lematizacion
       data = lemmatizer(data)

    if characters_specials == True: # Remover caracteres especiales
        data = remove_special_characters(data)

    if extra_whitespace == True: # Remover espacios en blanco
        data = remove_whitespace(data)    
       
    return data

# Función para realizar el proceso de detección de noticias falsas: 
# 1. Preprocesamiento del texto
# 2. Vectorización del texto - tfidf
# 3. Clasificación de la noticia
def detection_fake_news_(news,model,tfidf):
    news=text_preprocessing(news)
    input_data = [news]
    vectorized_input_data = tfidf.transform(input_data)
    prediction = model.predict(vectorized_input_data)
    prediction_proba = model.predict_proba(vectorized_input_data)
    if prediction==0:
      return "La noticia 📰 que ingreso es real con una probabilidad de: ",prediction_proba[0][0]
    else:
       return "La noticia 📰 que ingreso es falsa con una probabilidad de: ",prediction_proba[0][1]
