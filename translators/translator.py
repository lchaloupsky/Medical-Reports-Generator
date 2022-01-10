#!/usr/bin/python3

from requests.models import Response
from abc import ABC, abstractmethod

class Translator(ABC):
	@abstractmethod
	def translate(self, text: str) -> Response:
		'''Translates text.'''
		raise NotImplementedError()

	@abstractmethod
	def getText(self, text:str, response: Response) -> str:
		'''Recreates translated text from response.'''
		raise NotImplementedError()
