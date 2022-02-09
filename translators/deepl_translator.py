#!/usr/bin/python3

import requests
import json
import re
import random
import time
import copy

from requests.models import Response
from translators.translator import Translator

_proxies = ["5.189.184.6:80", "173.249.57.9:443", "138.201.120.214:1080"] #eval(open("filtered_p.txt", "r").readlines()[0])#
_proxy = 0

class _DeepLClientState():
	_url = 'https://w.deepl.com/web?request_type=jsonrpc&il=E&method=getClientState' #'https://www.deepl.com/PHP/backend/clientState.php?request_type=jsonrpc&il=en&method=getClientState'
	_body = {
		"id": 54421367,
		"jsonrpc": "2.0",
		"method": "getClientState",
		"params": {
			"v": "20180814"
		}
	}

	def get_state(self) -> Response:
		global _proxy
		while True:
			try:
				return requests.post(self._url, data=json.dumps(self._get_body()), headers={'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0'}, proxies={'https': 'http://' + _proxies[_proxy], 'http': 'http://' + _proxies[_proxy]}, timeout=10000)
			except Exception as e:
				print(e)
				_proxy = 0 if _proxy == len(_proxies) - 1 else _proxy + 1

	def _get_body(self) -> dict:
		body = self._body.copy()
		body["id"] = 10000 * round(random.random() * 1e4) #random.randrange(100, 10000) * 1000

		return body

class DeepLTranslator(Translator):
	_PAUSE = 8.0
	_url = 'https://www2.deepl.com/jsonrpc?method=LMT_handle_jobs'
	_body = {
		"jsonrpc": "2.0",
		"method": "LMT_handle_jobs",
		"params": {
			"jobs": [],
			"lang": {
				"preference": {
					"default": "default",
					"weight": {}
				},
				"source_lang_computed": "EN",
				"target_lang": "CS"
			},
			"priority": 1,
			"commonJobParams": {
				"browserType": 1,
				"formality": None
			},
			"timestamp": 1638706355399
		},
		"id": 39610005
	}
	_job_template = {
		"kind": "default",
		"preferred_num_beams": 1,
		"sentences": None,
		#"raw_en_sentence": "text",
		"raw_en_context_before": [],
		"raw_en_context_after": []
	}
	_headers = {
		'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0',
		'Referer': 'https://www.deepl.com/',
		'Origin': 'https://www.deepl.com',
		'authority': 'www2.deepl.com',
		'Accept-Language': 'cs,sk;q=0.8,en-US;q=0.5,en;q=0.3',
		'Content-Type': 'application/json',
		'Connection': 'keep-alive',
		'Accept': '*/*',
		'Accept-Encoding': 'gzip, deflate, br'
	}

	def __init__(self) -> None:
		super().__init__()
		self._client_state = _DeepLClientState()
		self._id_number = json.loads(self._client_state.get_state().text)["id"] #80030004
		self._last_request_time = 0
		self._request_count = 0

	def reset(self) -> None:
		global _proxy
		_proxy = 0 if _proxy == len(_proxies) - 1 else _proxy + 1 
		print(_proxy, self._request_count)
		self._client_state = _DeepLClientState()
		self._id_number = json.loads(self._client_state.get_state().text)["id"]

	def translate(self, text: str) -> Response:
		self._request_count += 1
		if self._request_count % 10 == 0:
			self.reset()

		# maybe do locking to be thread safe?
		while not self._can_send_request():
			# wait at least self._PAUSE seconds between consecutive requests
			time.sleep(abs(self._PAUSE + self._last_request_time - time.time()))
		
		reponse, retry_counter, failed_proxies = None, 0, 0
		while True:
			try:
				reponse = requests.post(self._url, data=json.dumps(self._fill_text(text)), headers=self._headers, proxies={'https': 'http://' + _proxies[_proxy], 'http': 'http://' + _proxies[_proxy]}, timeout=10000)
				failed_proxies = 0
				break
			except Exception as e:
				retry_counter += 1
				if retry_counter == 2:
					print(f"Proxy: {_proxy} failed. {e}")
					failed_proxies += 1
					retry_counter = 0
					self.reset()

			if failed_proxies == len(_proxies) - 1:
				print("All proxies failed.")
				break
		
		self._last_request_time = time.time()
		return reponse

	def get_text(self, text:str, response: Response) -> str:
		response = json.loads(response.text)
		return self._recreate_original_text(text, list(map(lambda x: x['beams'][0]['sentences'][0]['text'], response['result']['translations'])))

	def _can_send_request(self) -> bool:
		return time.time() - self._last_request_time >= self._PAUSE

	def _fill_text(self, text: str) -> dict:
		'''Splits text to translate into suitable DeepL parts and fills request object with them.'''
		splitted = self._flatten(self._split_into_parts(text))

		obj = copy.deepcopy(self._body)

		self._id_number += 1
		obj["id"] = self._id_number

		# this interesting timestamp construction is reverse engeneered from the site.
		#timestamp = int(time.time() * 10) * 100 + 1000
		timestamp = int(time.time() * 1000)
		sum_i = sum([s.count("i") for s in splitted]) + 1
		obj["params"]["timestamp"] = timestamp  + (sum_i - timestamp % sum_i) #1641208131535

		for idx in range(len(splitted)):
			job = self._job_template.copy()
			#job["raw_en_sentence"] = splitted[idx]
			job["sentences"] = [
				{
					"text": splitted[idx],
					"id": idx,
					"prefix": ""
				}
			]
			job["raw_en_context_before"] = splitted[max(0, idx - 5) :  idx]
			job["raw_en_context_after"] = splitted[idx + 1 : idx + 2]
			obj["params"]["jobs"].append(job)

		return obj

	def _recreate_original_text(self, text: str, translated: list[str]) -> str:
		'''Recreates original text using translated fragments.'''
		splitted = self._split_into_parts(text, use_strip=False)

		if len(translated) != len(list(filter(lambda x: x.strip(), self._flatten(splitted)))):
			print('Cannot fit translated into original. Number of parts does not correspond.')
			return None

		translated_iter = iter(translated)
		for line in splitted:
			if len(line) == 0: continue
			for i in range(len(line)):
				if not line[i].strip(): continue

				left_index = len(line[i]) - len(line[i].lstrip())
				right_index = len(line[i].rstrip())

				line[i] = line[i].replace(line[i][left_index : right_index], next(translated_iter))

		return "\n".join(["".join(s) for s in splitted])

	def _split_into_parts(self, text: str, use_strip: bool = True) -> list[list[str]]:
		'''Pipeline for splitting text into suitable DeepL parts.'''
		splitted_new_lines = text.split('\n')
		splitted_sentences = map(
			lambda x: list(re.split('([.|?|!|:]+)', x)), 
			splitted_new_lines
		)

		splitted = []
		for s in splitted_sentences:
			if len(s) % 2 != 0: s.append('')

			it = iter(s)
			splitted.append(list(
				filter(None, map(lambda x: (x[0] + x[1]).strip() if use_strip else x[0] + x[1], zip(it, it)))
			))

		return splitted

	def _flatten(self, list: list[str]) -> list[str]:
		'''Flattens a list of lists into a flat list with a filthy trick.'''
		return sum(list, [])
