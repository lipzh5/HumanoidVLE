# -*- coding:utf-8 -*-
# @Author: Peizhen Li 
# @Desc: None
import torch
from collections import deque
from CONF import frame_buffer_max_len, target_size, pretrained_path, encoding
from transformers import AutoTokenizer, RobertaTokenizer, RobertaModel, AutoImageProcessor
import torchvision.transforms as transforms
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
import time
from data_buffers import frame_buffer, diag_buffer



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



# ===========================↓↓↓ context modeling ↓↓↓==========================
def get_text_inputs_from_raw():
	query = 'For utterance:'
	query_ids = tokenizer(query)['input_ids'][1:-1]

	utterance_ids = []
	for idx, utt in enumerate(diag_buffer.dialogue):
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
		# print(f'len input ids: {len(input_ids)} \n &&&&&&&&&&&&&&&&&')
		input_ids, _ = pad_to_len(input_ids, max_len=context_max_len, pad_value=context_pad_value) # CONFIG['max_len'], CONFIG['pad_value']
	return torch.tensor(input_ids)
# ===========================↑↑↑ context modeling ↑↑↑==========================

def get_center_faces(img_arr):
	"""extract faces from raw image"""
	boxes, probs = face_detector.detect(img_arr)    # boxes: Nx4 array
	if boxes is None:
		return None
	box_order = np.argsort(np.abs((boxes[:, 2] + boxes[:, 0]) /2 - ORIGINAL_IMG_SHAPE[1]//2))  # [::-1]
	selected_boxes = boxes[0].reshape(-1, 4)
	faces = face_detector.extract(img_arr, selected_boxes, save_path=None)
	return faces

#TODO select faces of active speakers

def get_vision_inputs_from_raw(ts_end, duration):
	'''ref to: https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py'''
	
	n_frames = min(int(duration * FPS), MAX_FRAMES)   
	all_faces = []
	ref_face = None
	for i in range(n_frames, 0, -1):
		img_arr = np.asarray(Image.open(io.BytesIO(frame_buffer.buffer_content[-i])))
		# face_tensors = face_detector(img_arr)   # (n, 3, 160, 160)
		face_tensors = get_center_faces(img_arr)
		if face_tensors is not None:
			n_faces = face_tensors.shape[0]
			for i in range(n_faces):
				face = face_tensors[i]
				'''person-specific normalization'''
				if ref_face is None:
					ref_face = face
				face = face - ref_face  
				face = normalize(face)   # TODO apply normalization???
				all_faces.append(face)
	
	if all_faces:	
		all_faces = torch.stack(all_faces)
		all_faces, mask = pad_to_len(all_faces, MAX_FACES , pad_value=0)
		return all_faces, mask
	mask = torch.zeros([MAX_FACES,])
	mask[:2] = 1    # in case no real faces
	return torch.zeros([MAX_FACES, 3, 160, 160]), mask.long()
	# return np.concatenate(all_faces)  if all_faces else None



# ========================================================
# ↓↓↓ For TelME inference ↓↓↓
# ========================================================

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
speaker_list = ['<s1>', '<s2>', '<s3>', '<s4>', '<s5>', '<s6>', '<s7>', '<s8>', '<s9>']
speaker_tokens_dict = {'additional_special_tokens': speaker_list}
video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

def encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)
    
    return ids + [tokenizer.mask_token_id]

def padding(ids_list, tokenizer):
    max_len = 0
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    attention_masks = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        attention_mask = [ 1 for _ in range(len(ids))]
        add_attention = [ 0 for _ in range(len(add_ids))]
        pad_ids.append(add_ids+ids)
        attention_masks.append(add_attention+attention_mask)
    return torch.tensor(pad_ids), torch.tensor(attention_masks)

def get_text_inputs_from_raw_telme():
	input_str = ""
	for idx, utt in enumerate(diag_buffer.dialogue):
		input_str += "<s1>"    # we often do not have speaker identity in real world 
		input_str += f"<s1> {utt}"

	prompt = f"Now <s1> feels"
	concat_str = f"{input_str.strip()} </s> {prompt}"
	batch_input = [encode_right_truncated(concat_str, roberta_tokenizer)]
	batch_input_tokens, batch_attention_masks = padding(batch_input, roberta_tokenizer)
	return batch_input_tokens, batch_attention_masks
# --------------------------------------------------

def padding_video(batch):
    max_len = 0
    for ids in batch:
        if len(ids) > max_len:
            max_len = len(ids)
    
    pad_ids = []
    for ids in batch:
        pad_len = max_len-len(ids)
        add_ids = [ 0 for _ in range(pad_len)]
        
        pad_ids.append(add_ids+ids.tolist())
    return torch.tensor(pad_ids)

def get_vision_inputs_from_raw_telme(ts_end, duration):
	n_frames = min(int(duration * FPS), MAX_FRAMES)   
	frames = []
	step = n_frames // 8
	count = 0
	for i in range(n_frames, 0, -1):
		img_arr = np.asarray(Image.open(io.BytesIO(frame_buffer.buffer_content[-i])))
		count += 1
		if count % step == 0:
			frames.append(img_arr)

	if len(frames) >= 8:
		inputs = feature_extractor(frames[:8], return_tensors="pt")
		vision_inputs = inputs["pixel_values"][0]      
	else:
		lack = 8 - len(frames)
		extend_frames = [ frames[-1].copy() for _ in range(lack)]
		frames.extend(extend_frames)  
		inputs = feature_extractor(frames[:8], return_tensors="pt")
		vision_inputs = inputs["pixel_values"][0]
	return padding_video([vision_inputs])

# ========================================================
# ↑↑↑ For TelME inference ↑↑↑
# ========================================================


