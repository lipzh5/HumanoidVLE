# -*- coding:utf-8 -*-
# @Author: Peizhen Li 
# @Desc: None

import torch
from collections import deque
from CONF import diag_buffer_max_len, frame_buffer_max_len, target_size, pretrained_path, encoding
import time
# from utils import *

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
	dialogue = deque()

	def __new__(cls, *args, **kwargs):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance
	

	@classmethod
	def update_dialogue(cls, utterance):  # TODO add speaker info later
		while len(cls.dialogue) >= diag_buffer_max_len:
			cls.dialogue.popleft()
		cls.dialogue.append(utterance)
		print(f'dia buffer dialog :{len(cls.dialogue)}, utterance: {utterance} \n ****')
	
	@classmethod
	def clear_buffer(cls):
		cls.dialogue.clear()

	def __len__(self):
		return len(self.dialogue)
	# @classmethod
	# def reset(cls):
	# 	cls.dialogue = []


frame_buffer = FrameBuffer()
diag_buffer = DialogueBuffer()
