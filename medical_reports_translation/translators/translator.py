#!/usr/bin/python3

from requests.models import Response
from abc import ABC, abstractmethod

class Translator(ABC):
	'''Abstract class for text translators.'''

	@abstractmethod
	def translate(self, text: str) -> Response:
		'''
		Translates input text.
		
		:param text: Text to be translated
		:return: Translator service response object
		'''
		raise NotImplementedError()

	@abstractmethod
	def get_text(self, text: str, response: Response) -> str:
		'''
		Recreates translated text from response.
		
		:param text: Original report text
		:param response: Translator response from the translate() method
		:return: Translated text
		'''
		raise NotImplementedError()
