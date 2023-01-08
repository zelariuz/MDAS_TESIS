#!/usr/bin/python
# -*- coding: utf-8 -*-

#twisted
from twisted.enterprise import adbapi
#jsonrpc
from txjsonrpc.web import jsonrpc
#asyncDB twistar
# from twistar.registry import Registry
#Diferidos
from twisted.internet.defer import inlineCallbacks

#modules
# from library.models import EndpointModels

#Utilidades
from library.utils import decodeFile, encodeFile, createTestFile, loadModels, trainingModels, predictData

class ServerFactory(jsonrpc.JSONRPC):
	"""
		Back API Taller de Aplicaciones 2
	"""
	documentPath = ""

	def __init__(self, *args, **kwargs):
		"""
			Initial configuration for SherpaBackend jsonRPC server
			
		"""
		# inicializar caracteristicas del objeto JSONRPC
		super(jsonrpc.JSONRPC, self).__init__(*args, **kwargs)

	### Espacio de endpoints
	@inlineCallbacks
	def jsonrpc_alive(self):
		"""
			Para indicar si el reactor esta vivo.
		"""
		jsonResponse = {'response' : 1}
		yield 1+1 # condicion minima para que sea un diferido
		return jsonResponse

	@inlineCallbacks
	def jsonrpc_createBaseModel(self):
		"""
			Creamos el modelo base
		"""
		filename = 'model/model.models'
		trainingModels(filename)

		jsonResponse = {'response' : 1}
		yield 1+1 # condicion minima para que sea un diferido
		return jsonResponse

	@inlineCallbacks
	def jsonrpc_predict(self, data):
		"""
			Creamos el modelo base
		"""
		filename = 'model/model.models'
		models = loadModels(filename)

		predict = predictData(models, **data['EDI'], **data['DEP'])
		# result = model_lr.predict([text])

		# predict = {
		# 	'oneCluster' : {
		# 		'data' : [],
		# 		'forecast' : [],
		# 		'cluster' : 1,
		# 	},
		# 	'soubleCluster' : {
		# 		'data' : [],
		# 		'forecast' : [],
		# 		'clusterEdi' : 5,
		# 		'clusterDep' : 2,
		# 	}
		# }

		jsonResponse = {'response' : 1, 'predict': predict}
		yield 1+1 # condicion minima para que sea un diferido
		return jsonResponse