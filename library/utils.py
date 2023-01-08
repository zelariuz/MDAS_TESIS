import base64
import joblib
import pickle

## -------- Librerias de colab -------
import numpy as np
import json
import requests
import pandas as pd
from sklearn.cluster import KMeans
import math
# Cluster
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

#Series de Tiempo (Medias moviles)
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset
import gc


#### ------------ Variables globales ------
GLOBAL_LOADED = {} # {<fileName> : data}

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

def loadModels(filename):
	global GLOBAL_LOADED
	#Verificar que existe en memoria
	if (filename not in GLOBAL_LOADED):
		load = joblib.load(filename)
		GLOBAL_LOADED[filename] = load
	return GLOBAL_LOADED[filename]

def dbs_predict(db, x):
	dists = np.sqrt(np.sum((db.components_ - x)**2, axis=1))
	i = np.argmin(dists)
	return db.labels_[db.core_sample_indices_[i]] if dists[i] < db.eps else -1

### -------- Metodos del modelo ----------
def preProcess(s):
	return s

def trainingModels(filename):
	global GLOBAL_LOADED
	models = {}

	#Carga de datos Clean (colab)
	# {
	# 	'DEP_CARACTERISTICAS_CLEAN' : DEP_CARACTERISTICAS_CLEAN,
	# 	'EDI_CARACTERISTICAS_CLEAN' : EDI_CARACTERISTICAS_CLEAN,
	# 	'DEP_CARGOS_MENSUALES_CLEAR': DEP_CARGOS_MENSUALES_CLEAR,
	# }
	datas_dashboard = joblib.load('model/datas_dashboard.joblib')
	## Save datas
	DEP_CARACTERISTICAS_CLEAN = datas_dashboard['DEP_CARACTERISTICAS_CLEAN']
	EDI_CARACTERISTICAS_CLEAN = datas_dashboard['EDI_CARACTERISTICAS_CLEAN']
	DEP_CARGOS_MENSUALES_CLEAR = datas_dashboard['DEP_CARGOS_MENSUALES_CLEAR']

	##################################### DBS SCAN 
	##### DBS SCAN A (DEP)
	# Preparar Data
	DEP_DBSCAN = DEP_CARACTERISTICAS_CLEAN.copy()
	DEP_SUBTOTAL_AGG = DEP_CARGOS_MENSUALES_CLEAR.groupby(['EDIFICIO', 'DEPTO']).agg({'SUBTOTAL_GC': ['count', 'min', 'max', 'mean']})
	DEP_SUBTOTAL_AGG.reset_index(inplace=True)
	DEP_SUBTOTAL_AGG
	DEP_DBSCAN = pd.merge(DEP_DBSCAN, DEP_SUBTOTAL_AGG, how="left", on=["EDIFICIO", 'DEPTO'])
	DEP_DBSCAN

	#Entrenamiento
	DEP_DBSCAN_CLUSTER_COLS = ["PRORRATEO", "M_DEP", ('SUBTOTAL_GC', 'mean')]
	DEP_DBSCAN_CLUSTERS = DEP_DBSCAN[ DEP_DBSCAN_CLUSTER_COLS ]
	eps = 0.5
	DEP_DBSCAN_CLUSTERS_MODEL = DBSCAN(eps=eps, min_samples=4).fit(DEP_DBSCAN_CLUSTERS)

	##### DBS SCAN A (EDI)
	EDI_DBSCAN = EDI_CARACTERISTICAS_CLEAN.copy()
	# Juntar variables
	EDI_N_ESPACIOS_COLS = [ 'N_ESTACIONAMIENTO_VISITA', 'N_QUINCHOS', 'N_SALAS_REUNIONES', 'N_BICICLETERO', 'N_ACENSORES', 'N_ESTACIONAMIENTOS', 'N_BODEGA' ]
	EDI_DBSCAN[ 'N_ESPACIOS' ] = EDI_DBSCAN[ EDI_N_ESPACIOS_COLS ].sum(axis=1)
	EDI_N_CONSUMO_COLS = ['CONSUMO AGUA POTABLE', 'CONSUMO CALEFACCION', 'CONSUMO AGUA CALIENTE']
	EDI_DBSCAN[ 'N_CONSUMO' ] = EDI_DBSCAN[ EDI_N_CONSUMO_COLS ].sum(axis=1)
	## Quitamos variables ya juntadas
	EDI_DBSCAN = EDI_DBSCAN.drop(columns= EDI_N_ESPACIOS_COLS + EDI_N_CONSUMO_COLS)

	## Entrenamiento
	EDI_DBSCAN_CLUSTER_COLS = ['ENTREGA_OFICIAL', 'N_ESPACIOS', 'N_DEPTOS', 'M_TOTAL'] + ['N_CONSUMO']
	EDI_DBSCAN_CLUSTERS = EDI_DBSCAN[ EDI_DBSCAN_CLUSTER_COLS ]
	eps = 2574.070852191136
	EDI_DBSCAN_CLUSTERS_MODEL = DBSCAN(eps=eps, min_samples=4).fit(EDI_DBSCAN_CLUSTERS)

	#### DBS SCAN B (EDI-DEP)
	#Preparar data
	EDI_DEP_DBSCAN = pd.merge(DEP_DBSCAN, EDI_DBSCAN, how="left", on=["EDIFICIO"])

	#Entrenamiento
	EDI_DEP_DBSCAN_CLUSTER_COLS = DEP_DBSCAN_CLUSTER_COLS + EDI_DBSCAN_CLUSTER_COLS
	EDI_DEP_DBSCAN_CLUSTERS = EDI_DEP_DBSCAN[ EDI_DEP_DBSCAN_CLUSTER_COLS ]
	eps = 2.3
	EDI_DEP_DBSCAN_CLUSTERS_MODEL = DBSCAN(eps=eps, min_samples=4).fit(EDI_DEP_DBSCAN_CLUSTERS)

	### RESULTADOS
	# El data frame EDI_DEP_DBSCAN_CLUSTERS contiene la data de ambos
	RESULT_DBSCAN = EDI_DEP_DBSCAN.copy()
	# Agregamos los resultados por cada cluster, primera prueba A
	RESULT_DBSCAN['EDI_CLUSTER'] = EDI_DBSCAN_CLUSTERS_MODEL.fit_predict( RESULT_DBSCAN[ EDI_DBSCAN_CLUSTER_COLS ] )
	RESULT_DBSCAN['DEP_CLUSTER'] = DEP_DBSCAN_CLUSTERS_MODEL.fit_predict( RESULT_DBSCAN[ DEP_DBSCAN_CLUSTER_COLS ] )
	# Datos de la segunda prueba B
	RESULT_DBSCAN['EDI_DEP_CLUSTER'] = EDI_DEP_DBSCAN_CLUSTERS_MODEL.fit_predict( RESULT_DBSCAN[ EDI_DEP_DBSCAN_CLUSTER_COLS ] )
	# Nos quedamos con los valores principales
	RESULT_DBSCAN = RESULT_DBSCAN[ ['EDIFICIO', 'DEPTO', 'DEP_CLUSTER', 'EDI_CLUSTER', 'EDI_DEP_CLUSTER'] ]

	#Anclar resultados del entrenamineto anterior
	RESULT_DBSCAN_DEP_CARGOS_MENSUALES = DEP_CARGOS_MENSUALES_CLEAR.merge(RESULT_DBSCAN, how="left", on=["EDIFICIO", "DEPTO"])
	## Separar año y mes del GC
	RESULT_DBSCAN_DEP_CARGOS_MENSUALES['GC_MES'] = RESULT_DBSCAN_DEP_CARGOS_MENSUALES[ 'GC' ].apply(lambda x : x[4:])
	RESULT_DBSCAN_DEP_CARGOS_MENSUALES['GC_YEAR'] = RESULT_DBSCAN_DEP_CARGOS_MENSUALES[ 'GC' ].apply(lambda x : x[:-2])

	### RESULTADOS AÑO-MES (GC)

	# Clusters por separado (A - ['DEP_CLUSTER', 'EDI_CLUSTER'])
	RESULT_DBSCAN_GC_A = RESULT_DBSCAN_DEP_CARGOS_MENSUALES[ ['EDI_CLUSTER', 'DEP_CLUSTER', 'SUBTOTAL_GC', 'GC'] ].groupby(['DEP_CLUSTER', 'EDI_CLUSTER', 'GC']).agg({'SUBTOTAL_GC': ['count', 'min', 'max', 'mean', 'std']})
	RESULT_DBSCAN_GC_A.reset_index(inplace=True)

	# Clusters por separado (B - [EDI_DEP_CLUSTER'])
	RESULT_DBSCAN_GC_B = RESULT_DBSCAN_DEP_CARGOS_MENSUALES[ ['EDI_DEP_CLUSTER', 'SUBTOTAL_GC', 'GC'] ].groupby(['EDI_DEP_CLUSTER', 'GC']).agg({'SUBTOTAL_GC': ['count', 'min', 'max', 'mean', 'std']})
	RESULT_DBSCAN_GC_B.reset_index(inplace=True)

	#### RESULTADOS MES 

	# Clusters por separado (A - ['DEP_CLUSTER', 'EDI_CLUSTER'])
	RESULT_DBSCAN_GC_MES_A = RESULT_DBSCAN_DEP_CARGOS_MENSUALES[ ['DEP_CLUSTER', 'EDI_CLUSTER', 'SUBTOTAL_GC', 'GC_MES'] ].groupby(['DEP_CLUSTER', 'EDI_CLUSTER', 'GC_MES']).agg({'SUBTOTAL_GC': ['count', 'min', 'max', 'mean', 'std']})
	RESULT_DBSCAN_GC_MES_A.reset_index(inplace=True)

	# Clusters por separado (B - ['EDI_DEP_CLUSTER'])
	RESULT_DBSCAN_GC_MES_B = RESULT_DBSCAN_DEP_CARGOS_MENSUALES[ ['EDI_DEP_CLUSTER', 'SUBTOTAL_GC', 'GC_MES'] ].groupby(['EDI_DEP_CLUSTER', 'GC_MES']).agg({'SUBTOTAL_GC': ['count', 'min', 'max', 'mean', 'std']})
	RESULT_DBSCAN_GC_MES_B.reset_index(inplace=True)

	#### Resultado ALL TIME

	# Clusters por separado (A - ['DEP_CLUSTER', 'EDI_CLUSTER'])
	RESULT_DBSCAN_A = RESULT_DBSCAN_DEP_CARGOS_MENSUALES[ ['DEP_CLUSTER', 'EDI_CLUSTER', 'SUBTOTAL_GC'] ].groupby(['DEP_CLUSTER', 'EDI_CLUSTER']).agg({'SUBTOTAL_GC': ['count', 'min', 'max', 'mean', 'std']})
	RESULT_DBSCAN_A.reset_index(inplace=True)

	# Clusters por separado (B - ['EDI_DEP_CLUSTER'])
	RESULT_DBSCAN_B = RESULT_DBSCAN_DEP_CARGOS_MENSUALES[ ['EDI_DEP_CLUSTER', 'SUBTOTAL_GC'] ].groupby(['EDI_DEP_CLUSTER']).agg({'SUBTOTAL_GC': ['count', 'min', 'max', 'mean', 'std']})
	RESULT_DBSCAN_B.reset_index(inplace=True)


	###################################### SERIES DE TIEMPO
	ST_CARGOS_MENSUALES_A = RESULT_DBSCAN_GC_A.copy()
	ST_CARGOS_MENSUALES_B = RESULT_DBSCAN_GC_B.copy()

	###### SERIES A
	SERIE_MM_A_MODELS = {}

	#Colector de basura
	gc.collect()

	for cluster, a in ST_CARGOS_MENSUALES_A.groupby(['DEP_CLUSTER', 'EDI_CLUSTER']):
	  title = f'Cluster Dep {cluster[0]} Edi {cluster[1]}'
	  status = 'OK'

	  # Ordenamos por GC (ASC)
	  a = a.sort_values(by=['GC'], ascending=True)

	  # Convertimos el index en la fecha del mes y nos quedamos con el unico valor del promedio GC
	  b = pd.DataFrame()
	  b['GC'] = a['GC'].apply(lambda x : x[:-2] +'-'+ x[4:] )
	  b['SUBTOTAL_GC_MEAN'] = a[('SUBTOTAL_GC', 'mean')]

	  b.index = pd.to_datetime(b['GC'])
	  b.drop(columns='GC',inplace=True)
	  b.sort_index(inplace=True)

	  #Serie original
	  error_previous_month = [0]
	  error1 = False
	  for i in range(1, len(b.index)):
	    _t = (b.index[i] - b.index[i-1]).days > 31
	    error_previous_month.append(int(_t))
	    error1 = error1 or _t

	  # Error por falta de datos (minimo 24)
	  error2 = (b.count().sum() < 24)

	  if (error1 or error2):
	    status = 'MES FALTANTE' if (error1 and not error2) else ('POCOS DATOS' if (error2 and not error1) else 'POCOS DATOS y MES FALTANTE')
	    #Save model ERROR
	    SERIE_MM_A_MODELS[cluster] = {
	        'DEP_CLUSTER' : cluster[0],
	        'EDI_CLUSTER' : cluster[1],
	        'model' : None,
	        'df' : b,
	        'status' : status,
	    }
	    continue

	  model=sm.tsa.statespace.SARIMAX(b['SUBTOTAL_GC_MEAN'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
	  results=model.fit()

	  # Predecir desde hace 12 meses
	  end = b['SUBTOTAL_GC_MEAN'].count()
	  start = end - 12

	  b['forecast']=results.predict(start=start, end=end,dynamic=True)

	  #Save model
	  SERIE_MM_A_MODELS[cluster] = {
	    'DEP_CLUSTER' : cluster[0],
	    'EDI_CLUSTER' : cluster[1],
	    'model' : results,
	    'df' : b,
	    'status' : status,
	  }

	  #break


	###### SERIES B
	#SERIE_MM_B_MODELS = { 'EDI_DEP_CLUSTER' : {'model' : Series Moviles, 'df' : Pandas, 'status' : <OK> o <POCOS DATOS> o <MES FALTANTE> o <POCOS DATOS y MES FALTANTE>} }
	SERIE_MM_B_MODELS = {}

	#Colector de basura
	gc.collect()

	for cluster, a in ST_CARGOS_MENSUALES_B.groupby(['EDI_DEP_CLUSTER']):
	  title = f'Cluster Edi-Dep {cluster}'
	  status = 'OK'

	  # Ordenamos por GC (ASC)
	  a = a.sort_values(by=['GC'], ascending=True)

	  # Convertimos el index en la fecha del mes y nos quedamos con el unico valor del promedio GC
	  b = pd.DataFrame()
	  b['GC'] = a['GC'].apply(lambda x : x[:-2] +'-'+ x[4:] )
	  b['SUBTOTAL_GC_MEAN'] = a[('SUBTOTAL_GC', 'mean')]

	  b.index = pd.to_datetime(b['GC'])
	  b.drop(columns='GC',inplace=True)
	  b.sort_index(inplace=True)

	  #Serie original
	  error_previous_month = [0]
	  error1 = False
	  for i in range(1, len(b.index)):
	    _t = (b.index[i] - b.index[i-1]).days > 31
	    error_previous_month.append(int(_t))
	    error1 = error1 or _t

	  # Error por falta de datos (minimo 24)
	  error2 = (b.count().sum() < 24)

	  if (error1 or error2):
	    status = 'MES FALTANTE' if (error1 and not error2) else ('POCOS DATOS' if (error2 and not error1) else 'POCOS DATOS y MES FALTANTE')
	    #Save model ERROR
	    SERIE_MM_B_MODELS[cluster] = {
	        'EDI_DEP_CLUSTER' : cluster,
	        'model' : None,
	        'df' : b,
	        'status' : status,
	    }
	    continue

	  model=sm.tsa.statespace.SARIMAX(b['SUBTOTAL_GC_MEAN'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
	  results=model.fit()

	  # Predecir desde hace 12 meses
	  end = b['SUBTOTAL_GC_MEAN'].count()
	  start = end - 12

	  b['forecast']=results.predict(start=start, end=end,dynamic=True)

	  #Save model
	  SERIE_MM_B_MODELS[cluster] = {
	    'EDI_DEP_CLUSTER' : cluster,
	    'model' : results,
	    'df' : b,
	    'status' : status,
	  }

	  #break

	models = {
		'RESULT_DBSCAN_A':RESULT_DBSCAN_A,
		'RESULT_DBSCAN_B':RESULT_DBSCAN_B,
		'RESULT_DBSCAN_GC_MES_A':RESULT_DBSCAN_GC_MES_A,
		'RESULT_DBSCAN_GC_MES_B':RESULT_DBSCAN_GC_MES_B,
		'RESULT_DBSCAN_GC_A':RESULT_DBSCAN_GC_A,
		'RESULT_DBSCAN_GC_B':RESULT_DBSCAN_GC_B,
		'RESULT_DBSCAN_DEP_CARGOS_MENSUALES':RESULT_DBSCAN_DEP_CARGOS_MENSUALES,
		'EDI_DBSCAN_CLUSTERS_MODEL':EDI_DBSCAN_CLUSTERS_MODEL,
		'DEP_DBSCAN_CLUSTERS_MODEL':DEP_DBSCAN_CLUSTERS_MODEL,
		'EDI_DEP_DBSCAN_CLUSTERS_MODEL':EDI_DEP_DBSCAN_CLUSTERS_MODEL,
		'SERIE_MM_A_MODELS':SERIE_MM_A_MODELS,
		'SERIE_MM_B_MODELS':SERIE_MM_B_MODELS,
	}

	# Guardado en memoria
	GLOBAL_LOADED[filename] = models
	## Guardado del modelo en un binario
	joblib.dump(models, filename)

def predictData(models, ENTREGA_OFICIAL=0, N_ESPACIOS=0, N_DEPTOS=0, M_TOTAL=0, N_CONSUMO=0,PRORRATEO=0, M_DEP=0, GC_MEAN=0):
	print({'ENTREGA_OFICIAL':ENTREGA_OFICIAL,'N_ESPACIOS':N_ESPACIOS,'N_DEPTOS':N_DEPTOS,'M_TOTAL':M_TOTAL,'N_CONSUMO,PRORRATEO':N_CONSUMO,'PRORRATEO':PRORRATEO,'M_DEP':M_DEP,'GC_MEAN':GC_MEAN})
	
	# EDI_DBSCAN_CLUSTERS_MODEL.fit_predict( _RESULT_DBSCAN[ EDI_DBSCAN_CLUSTER_COLS ] )
	# DEP_DBSCAN_CLUSTERS_MODEL.fit_predict( _RESULT_DBSCAN[ DEP_DBSCAN_CLUSTER_COLS ] )
	# EDI_DEP_DBSCAN_CLUSTERS_MODEL.fit_predict( _RESULT_DBSCAN[ EDI_DEP_DBSCAN_CLUSTER_COLS ] )
	EDI_DBSCAN_CLUSTERS_MODEL = models['EDI_DBSCAN_CLUSTERS_MODEL']
	DEP_DBSCAN_CLUSTERS_MODEL = models['DEP_DBSCAN_CLUSTERS_MODEL']
	EDI_DEP_DBSCAN_CLUSTERS_MODEL = models['EDI_DEP_DBSCAN_CLUSTERS_MODEL']

	RESULT_DBSCAN_A = models['RESULT_DBSCAN_A']
	RESULT_DBSCAN_GC_MES_A = models['RESULT_DBSCAN_GC_MES_A']
	RESULT_DBSCAN_GC_A = models['RESULT_DBSCAN_GC_A']
	SERIE_MM_A_MODELS = models['SERIE_MM_A_MODELS']

	RESULT_DBSCAN_B = models['RESULT_DBSCAN_B']
	RESULT_DBSCAN_GC_MES_B = models['RESULT_DBSCAN_GC_MES_B']
	RESULT_DBSCAN_GC_B = models['RESULT_DBSCAN_GC_B']
	SERIE_MM_B_MODELS = models['SERIE_MM_B_MODELS']

	outData = {
	    'oneCluster' : {},
	    'twoCluster' : {},
	}

	# ["PRORRATEO", "M_DEP", ('SUBTOTAL_GC', 'mean')]
	_DEP_DBSCAN_CLUSTER_COLS = [PRORRATEO, M_DEP, GC_MEAN]
	# ['ENTREGA_OFICIAL', 'N_ESPACIOS', 'N_DEPTOS', 'M_TOTAL'] + ['N_CONSUMO']
	_EDI_DBSCAN_CLUSTER_COLS = [ENTREGA_OFICIAL, N_ESPACIOS, N_DEPTOS, M_TOTAL] + [N_CONSUMO]
	_EDI_DEP_DBSCAN_CLUSTER_COLS = _DEP_DBSCAN_CLUSTER_COLS + _EDI_DBSCAN_CLUSTER_COLS
	print(_EDI_DEP_DBSCAN_CLUSTER_COLS)
	EDI_CLUSTER = dbs_predict(EDI_DBSCAN_CLUSTERS_MODEL, np.array(_EDI_DBSCAN_CLUSTER_COLS) )
	DEP_CLUSTER = dbs_predict(DEP_DBSCAN_CLUSTERS_MODEL, np.array(_DEP_DBSCAN_CLUSTER_COLS) )
	EDI_DEP_CLUSTER = dbs_predict(EDI_DEP_DBSCAN_CLUSTERS_MODEL, np.array(_EDI_DEP_DBSCAN_CLUSTER_COLS) )
	print(EDI_CLUSTER, DEP_CLUSTER, EDI_DEP_CLUSTER)

	outData['twoCluster']['EDI_CLUSTER'] = int(EDI_CLUSTER)
	outData['twoCluster']['DEP_CLUSTER'] = int(DEP_CLUSTER)
	outData['oneCluster']['EDI_DEP_CLUSTER'] = int(EDI_DEP_CLUSTER)

	# Obtener la data
	# Cluster A
	_tmp = RESULT_DBSCAN_A[ (RESULT_DBSCAN_A['DEP_CLUSTER'] == DEP_CLUSTER) & (RESULT_DBSCAN_A['EDI_CLUSTER'] == EDI_CLUSTER) ].to_json(orient="records")
	_tmp = json.loads(_tmp)
	outData['twoCluster']['gc_mean'] = _tmp[0] if (len(_tmp) > 0) else None

	_tmp = RESULT_DBSCAN_GC_MES_A[ (RESULT_DBSCAN_GC_MES_A['DEP_CLUSTER'] == DEP_CLUSTER) & (RESULT_DBSCAN_GC_MES_A['EDI_CLUSTER'] == EDI_CLUSTER) ].to_json(orient="records")
	_tmp = json.loads(_tmp)
	outData['twoCluster']['gc_month'] = _tmp

	_tmp = RESULT_DBSCAN_GC_A[ (RESULT_DBSCAN_GC_A['DEP_CLUSTER'] == DEP_CLUSTER) & (RESULT_DBSCAN_GC_A['EDI_CLUSTER'] == EDI_CLUSTER) ]
	_tmp['GC'] = _tmp['GC'].apply(lambda x : x[:4]+'-'+x[4:])
	_tmp = _tmp.to_json(orient="records")
	_tmp = json.loads(_tmp)
	outData['twoCluster']['gc'] = _tmp

	outData['twoCluster']['forecast'] = None
	if (DEP_CLUSTER,EDI_CLUSTER) in SERIE_MM_A_MODELS:
	  # Captura de la data de serie
	  _target = SERIE_MM_A_MODELS[(DEP_CLUSTER,EDI_CLUSTER)]
	  _df = _target['df']
	  _model = _target['model']

	  if (_model is not None):
	    # Predecir desde hace 12 meses
	    end = _df['SUBTOTAL_GC_MEAN'].count()
	    start = end - 12

	    # Predecir al futuro
	    pred_date=[_df.index[-1] + DateOffset(months=x) for x in range(0, 12)]
	    pred_date=pd.DataFrame(index=pred_date[1:], columns=_df.columns)
	    n=pd.concat([_df,pred_date])
	    n['forecast_year'] = _model.predict(start = end-1, end = end + 13, dynamic= True)  
	    #n[['SUBTOTAL_GC_MEAN', 'forecast_year']].plot(figsize=(14.5, 2))

	    n['month'] = n.index.strftime('%Y-%m')
	    outData['twoCluster']['forecast'] = json.loads(n.to_json(orient="records"))

	# Cluster B
	_tmp = RESULT_DBSCAN_B[ RESULT_DBSCAN_B['EDI_DEP_CLUSTER'] == EDI_DEP_CLUSTER ].to_json(orient="records")
	_tmp = json.loads(_tmp)
	outData['oneCluster']['gc_mean'] = _tmp[0] if (len(_tmp) > 0) else None

	_tmp = RESULT_DBSCAN_GC_MES_B[ RESULT_DBSCAN_GC_MES_B['EDI_DEP_CLUSTER'] == EDI_DEP_CLUSTER ].to_json(orient="records")
	_tmp = json.loads(_tmp)
	outData['oneCluster']['gc_month'] = _tmp

	_tmp = RESULT_DBSCAN_GC_B[ RESULT_DBSCAN_GC_B['EDI_DEP_CLUSTER'] == EDI_DEP_CLUSTER ]
	_tmp['GC'] = _tmp['GC'].apply(lambda x : x[:4]+'-'+x[4:])
	_tmp = _tmp.to_json(orient="records")
	_tmp = json.loads(_tmp)
	outData['oneCluster']['gc'] = _tmp

	outData['oneCluster']['forecast'] = None
	if EDI_DEP_CLUSTER in SERIE_MM_B_MODELS:
	  # Captura de la data de serie
	  _target = SERIE_MM_B_MODELS[EDI_DEP_CLUSTER]
	  _df = _target['df']
	  _model = _target['model']

	  if (_model is not None):
	    # Predecir desde hace 12 meses
	    end = _df['SUBTOTAL_GC_MEAN'].count()
	    start = end - 12

	    # Predecir al futuro
	    pred_date=[_df.index[-1] + DateOffset(months=x) for x in range(0, 12)]
	    pred_date=pd.DataFrame(index=pred_date[1:], columns=_df.columns)
	    n=pd.concat([_df,pred_date])
	    n['forecast_year'] = _model.predict(start = end-1, end = end + 13, dynamic= True)  
	    #n[['SUBTOTAL_GC_MEAN', 'forecast_year']].plot(figsize=(14.5, 2))

	    n['month'] = n.index.strftime('%Y-%m')
	    outData['oneCluster']['forecast'] = json.loads(n.to_json(orient="records"))

	return outData