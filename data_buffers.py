# -*- coding:utf-8 -*-
# @Author: Peizhen Li 
# @Desc: None

import torch
from collections import deque
from CONF import frame_buffer_max_len, target_size, pretrained_path, encoding
import time
from utils import tokenizer, pad_to_len


class FrameBuffer:
	_instance = None
	buffer_content = deque()
	arrival_time = 0     # fps = 25 (~0.04s / Frame)

	def __new__(cls, *args, **kwargs):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance
	
	def __len__(self):
		return len(self.buffer_content)

	@classmethod
	def append_content(cls, frame: bytes):
		while len(cls.buffer_content) >= frame_buffer_max_len:
			cls.buffer_content.popleft()
		cls.buffer_content.append(frame)
		cls.arrival_time = time.time()

	@classmethod
	def consume_content(cls, num):
		if num > len(cls.buffer_content):
			return None
		return [cls.buffer_content.popleft() for _ in range(num)]

	@classmethod
	def consume_one_frame(cls):
		if cls.buffer_content:
			return cls.buffer_content[-1]
			# return cls.buffer_content.popleft()
		return None


class DialogueBuffer:
	_instance = None
	dialogue = []

	def __new__(cls, *args, **kwargs):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance
	

	@classmethod
	def update_dialogue(cls, utterance):  # TODO add speaker info later
		cls.dialogue.append(utterance)
		print(f'dia buffer dialog :{len(cls.dialogue)}, utterance: {utterance} \n ****')

	@classmethod
	def reset(cls):
		cls.dialogue = []

	@classmethod
	def get_text_inputs_ids(cls):
		query = 'For utterance:'
		query_ids = tokenizer(query)['input_ids'][1:-1]

		utterance_ids = []
		for idx, utt in enumerate(cls.dialogue):
			token_ids = tokenizer(utt.decode(encoding))['input_ids'][1:]
			utterance_ids.append(token_ids)
			full_context = [CLS]
			lidx = 0
			for lidx in range(idx):
				total_len = sum([len(item) for item in utterance_ids[lidx:]]) + 8
				if total_len + len(utterance_ids[idx]) <= context_max_len: # CONFIG['max_len']:
					break
			lidx = max(lidx, idx-8)
			for item in utterance_ids[lidx:]:
				full_context.extend(item)

			query_idx = idx
			# prompt = dialogue[query_idx]['speaker'] + ' feels <mask>'
			prompt = 'speaker feels <mask>'
			full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:]
			input_ids = full_context + full_query
			print(f'len input ids: {len(input_ids)} \n &&&&&&&&&&&&&&&&&')
			input_ids, _ = pad_to_len(input_ids, max_len=context_max_len, pad_value=context_pad_value) # CONFIG['max_len'], CONFIG['pad_value']
		return torch.tensor(input_ids)
		 
frame_buffer = FrameBuffer()
diag_buffer = DialogueBuffer()
