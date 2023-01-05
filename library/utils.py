import base64
import joblib
import pickle

## -------- Librerias de colab -------
#Ingesta
import pandas as pd

#Preprocesamiento
import numpy as np

#Procesamiento Lenguaje Natural
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

stopWords = set(stopwords.words('english'))

#Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Codificacion
from sklearn.preprocessing import LabelEncoder

#Grilla y Pipeline
# from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

#Muestras
from sklearn.model_selection import train_test_split

#Modelamiento Predictivo
from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier


def encodeFile(filename):
	base64_message = None
	with open(filename, 'rb') as binary_file:
		binary_file_data = binary_file.read()
		base64_encoded_data = base64.b64encode(binary_file_data)
		base64_message = base64_encoded_data.decode('utf-8')
	return base64_message

def decodeFile(filename, textBase64):
	base64_bytes = textBase64.encode('utf-8')
	with open(filename, 'wb') as file_to_save:
		decoded_image_data = base64.decodebytes(base64_bytes)
		file_to_save.write(decoded_image_data)

def _obj(x):
	return x

def createTestFile(filename):
	obj = ['hola', 'Exito']
	# pickle.dump(_obj, open(filename,'wb'))
	joblib.dump(_obj, filename)

def loadModelObjectFile(filename):
	return joblib.load(filename)
	# return pickle.load(open(filename,'rb'))


### -------- Metodos del modelo ----------
def preProcess(s):

  """Description of the Function
  Esta funcion preprocesa texto escrito en lenguaje natural idioma español

  Parameters:
  s: string con el texto

  Returns:
  devuelve el string limpio con las funciones de NLP

 """

  lemmatizer = WordNetLemmatizer()
  stemmer = PorterStemmer()

  string = s.lower() # convierte a minusculas
  string = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', string) # remueve URLs
  string = re.sub('@[^\s]+', '', string) # remueve nombres de usuario
  string = re.sub('[^a-zA-Z \n\.]', '', string) # remueve numeros y caracteres especiales
  string = re.sub(r'\b(\w+)( \1\b)+', r'\1', string) # remueve palabras repetidas

  token = nltk.word_tokenize(string)

  lemmatized_output = ' '.join([w for w in token if w.isalpha()])

  token1 = nltk.word_tokenize(lemmatized_output)
  lemmatized_output1 = ' '.join([w for w in token1 if len(w) > 1])

  token2 = nltk.word_tokenize(lemmatized_output1)
  lemmatized_output2 = ' '.join([w for w in token2 if len(w) < 8])

  token3 = nltk.word_tokenize(lemmatized_output2)
  lemmatized_output3 = ' '.join([lemmatizer.lemmatize(w) for w in token3])

  token4 = nltk.word_tokenize(lemmatized_output3)
  lemmatized_output4 = ' '.join([stemmer.stem(w) for w in token4])

  token5 = nltk.word_tokenize(lemmatized_output4)
  lemmatized_output5 = ' '.join([w for w in token5 if not w in stopWords])

  #token6 = nltk.word_tokenize(lemmatized_output5)
  #lemmatized_output6 = ' '.join([(''.join([j for i,j in enumerate(w) if j not in w[:i]])) for w in token6])

  #print(lemmatized_output6)

  return lemmatized_output5

def trainingModelBase(filename):
	df = pd.read_csv('model/training_tweets.csv').drop(columns= 'Unnamed: 0')
	## Quedarnos con los NO neutrales
	df = df.loc[df['sentiment'] != 'neutral']
	df = df.reset_index(drop=True)

	## Etiquetado
	Negativa = ['worry','sadness','hate','empty','boredom','anger']
	Positiva = ['happiness','love','surprise','fun','relief','enthusiasm']
	
	conditions  = [df['sentiment'].isin(Negativa), df['sentiment'].isin(Positiva)]
	choices     = ['Negativa','Positiva']

	df['sentiment_recod'] = np.select(conditions, choices)

	## Entrenamiento
	lbl_label = LabelEncoder()
	df["sentiment_recod"] = lbl_label.fit_transform(df["sentiment_recod"])

	## Creando el modelo y entrenandolo
	x_train, x_test, y_train, y_test = train_test_split(df['content'], df['sentiment_recod'] ,test_size=.33, random_state=3645)

	pipeline_lr = Pipeline([('tokenizar', CountVectorizer(preprocessor=preProcess,ngram_range=(1, 2))),
                        ('clasificador', LogisticRegression())])

	model_lr = pipeline_lr.fit(x_train,y_train);

	## Guardado del modelo en un binario
	joblib.dump(model_lr, filename)