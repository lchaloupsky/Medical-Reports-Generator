#!/usr/bin/python3

import requests

from requests.models import Response
from translators.translator import Translator

class CubbittTranslator(Translator):
	'''Class for translating texts using the CUBBITT translator.'''
	_url = 'https://lindat.mff.cuni.cz/services/translation/api/v2/models/'
	_body = {
		'input_text': 'text',
		'src': 'en',
		'tgt': 'cs'
	}

	def __init__(self, model: str = "doc-en-cs"):
		'''
		Constructs new CubbittTranslator instance.
		
		:param model: Name of the model to bed used for the translation
		'''
		super().__init__()
		self._model = model
		self._src_lang = model.replace("doc-", "").split("-")[0]
		self._tgt_lang = model.replace("doc-", "").split("-")[1]

	def translate(self, text: str) -> Response:
		return requests.post(self._create_url(), data=self._fill_text(text))

	def get_text(self, _:str, response: Response) -> str:
		return response.text[:-1]

	def _create_url(self):
		return self._url + self._model

	def _fill_text(self, text: str) -> object:
		obj = self._body.copy()
		obj["input_text"] = text
		obj["src"] = self._src_lang
		obj["tgt"] = self._tgt_lang
		
		return obj
