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
from library.utils import decodeFile, encodeFile, createTestFile, loadModelObjectFile, trainingModelBase

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
	def jsonrpc_testEncodeFile(self):
		"""
			Para indicar si el reactor esta vivo.
		"""
		createTestFile('model/test_load.model')
		modelDataBase64 = encodeFile('model/test_load.model')

		jsonResponse = {'response' : 1, 'modelDataBase64' : modelDataBase64}
		yield 1+1 # condicion minima para que sea un diferido
		return jsonResponse

	@inlineCallbacks
	def jsonrpc_uploadModelFileData(self, modelDataBase64 = ''):
		"""
			Subida del modelo en base 64
		"""
		print('modelDataBase64', modelDataBase64)
		decodeFile('model/test.model', modelDataBase64)
		f = loadModelObjectFile('model/test.model')
		print('loadModelObjectFile', f)

		jsonResponse = {'response' : 1, 'modelDataBase64' : modelDataBase64}
		yield 1+1 # condicion minima para que sea un diferido
		return jsonResponse

	@inlineCallbacks
	def jsonrpc_createBaseModel(self):
		"""
			Creamos el modelo base
		"""
		filename = 'model/modelBase.model'
		trainingModelBase(filename)

		jsonResponse = {'response' : 1}
		yield 1+1 # condicion minima para que sea un diferido
		return jsonResponse

	@inlineCallbacks
	def jsonrpc_predict(self, text = ''):
		"""
			Creamos el modelo base
		"""
		filename = 'model/modelBase.model'
		model_lr = loadModelObjectFile(filename)

		result = model_lr.predict([text])

		jsonResponse = {'response' : 1, 'predict': ('Positive' if result[0] == 1 else 'Negative')}
		yield 1+1 # condicion minima para que sea un diferido
		return jsonResponse