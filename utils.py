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


