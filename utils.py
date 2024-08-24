# -*- coding:utf-8 -*-
# @Author: Peizhen Li 
# @Desc: None
import torch
from collections import deque
from CONF import frame_buffer_max_len, target_size, pretrained_path, encoding
from transformers import AutoTokenizer
import torchvision.transforms as transforms
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
import time


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


class Transform:
	def __init__(self):  # cfg: config.data.transform
		self.transform = transforms.Compose(
			[transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])  

	def __call__(self, data):
		return self.transform(data)


class Resize:
	def __init__(self, target_size):
		self.target_size = target_size  # cfg.resize.target_size

	def __call__(self, img: np.ndarray):
		interp = cv2.INTER_AREA if img.shape[0] > self.target_size else cv2.INTER_CUBIC
		return cv2.resize(img, dsize=(self.target_size, self.target_size), interpolation=interp)

class Normalize:
	def __init__(self):
		self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

	def __call__(self, data):
		return self.normalize(data)

transform = Transform()
resize = Resize(target_size)
normalize = Normalize()


# tokenizer = None
# CONTEXT_CONF = {}
# def get_tokenizer(pretrained_path):
# 	global tokenizer
# 	if tokenizer is None:
# 		tokenizer = AutoTokenizer.from_pretrained(pretrained_path, local_files_only=False)
# 		_special_tokens_ids = tokenizer('<mask>')['input_ids']
# 		CLS = _special_tokens_ids[0]
# 		MASK = _special_tokens_ids[1]
# 		SEP = _special_tokens_ids[2]
# 		CONTEXT_CONF['CLS'] = CLS  
# 		CONTEXT_CONF['SEP'] = SEP
# 		CONTEXT_CONF['mask_value'] = MASK
# 	return tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_path, local_files_only=False)
_special_tokens_ids = tokenizer('<mask>')['input_ids']
CLS = _special_tokens_ids[0]
MASK = _special_tokens_ids[1]
SEP = _special_tokens_ids[2]

context_max_len = 256  
context_pad_value = 1  
# CONTEXT_CONF['CLS'] = CLS  
# CONTEXT_CONF['SEP'] = SEP
# CONTEXT_CONF['mask_value'] = MASK


def pad_to_len(sequence_data, max_len, pad_value):
	sequence_data = sequence_data[-max_len:]
	effective_len = len(sequence_data)
	mask = torch.zeros((max_len,))
	mask[:effective_len] = 1

	len_to_pad = max_len - effective_len
	
	if isinstance(sequence_data, list):
		pads = [pad_value]*len_to_pad
		sequence_data.extend(pads)
	elif isinstance(sequence_data, torch.Tensor):
		pads = torch.ones([len_to_pad, *sequence_data.shape[1:]]) * pad_value
		sequence_data = torch.concat((sequence_data, pads))
   
	return sequence_data, mask


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



def set_vision_encoder(cfg):
	if cfg.model.vision_encoder.model_name == 'inceptionresnetv1':
		use_webface_pretrain = cfg.model.vision_encoder.use_webface_pretrain
		print(f'Inception uses webface pretrain: {use_webface_pretrain} \n ******')
		vision_encoder = InceptionResnetV1(pretrained='casia-webface') if use_webface_pretrain else InceptionResnetV1()
	else:
		# from keras.applications import ResNet50
		use_imgnet_pretrain = cfg.model.vision_encoder.use_imgnet_pretrain
		# vision_encoder = ResNet50(include_top=False, weights='imagenet') if use_imgnet_pretrain else ResNet50(include_top=False) 
		from torchvision.models import resnet50, ResNet50_Weights
		vision_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) if use_imgnet_pretrain else resnet50()
		print(f'Resnet50 uses imgnet pretrain: {use_imgnet_pretrain} \n ******')

	vis_enc_trainable = cfg.train.resnet_trainable
	vision_encoder.train(vis_enc_trainable)
	vision_encoder.requires_grad_(vis_enc_trainable)
	return vision_encoder


