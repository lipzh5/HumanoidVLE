import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import random
import math
import pandas as pd
import os.path as osp
from utils import *
from models.emotion_rec import emo_recognizer


from transformers import RobertaTokenizer, RobertaModel, TimesformerModel, Data2VecAudioModel

class Teacher_model(nn.Module):
	def __init__(self, text_model, clsNum):
		super(Teacher_model, self).__init__()
		
		"""Text Model"""
		tmodel_path = text_model
		if text_model == 'roberta-large':
			self.text_model = RobertaModel.from_pretrained(tmodel_path)
			tokenizer = RobertaTokenizer.from_pretrained(tmodel_path)
			self.speaker_list = ['<s1>', '<s2>', '<s3>', '<s4>', '<s5>', '<s6>', '<s7>', '<s8>', '<s9>']
			self.speaker_tokens_dict = {'additional_special_tokens': self.speaker_list}
			tokenizer.add_special_tokens(self.speaker_tokens_dict)
			
		self.text_model.resize_token_embeddings(len(tokenizer))
		self.text_hiddenDim = self.text_model.config.hidden_size
		
		"""Logit"""
		self.W = nn.Linear(self.text_hiddenDim, 768)
		self.classifier = nn.Linear(768, clsNum)

	def forward(self, batch_input_tokens, attention_masks):

		batch_context_output = self.text_model(batch_input_tokens, attention_masks).last_hidden_state[:,-1,:] # (batch, 1024)
		
		batch_last_hidden = self.W(batch_context_output)
		context_logit = self.classifier(batch_last_hidden)

		return batch_last_hidden, context_logit



class Student_Audio(nn.Module):
	def __init__(self, audio_model, clsNum, init_config):
		super(Student_Audio, self).__init__()
		
		"""Model Setting"""
		amodel_path = audio_model
		if audio_model == "facebook/data2vec-audio-base-960h":
			
			self.model = Data2VecAudioModel.from_pretrained(amodel_path)
			self.model.config.update(init_config.__dict__)

		self.hiddenDim = self.model.config.hidden_size
			
		"""score"""
		self.W = nn.Linear(self.hiddenDim, clsNum)

	def forward(self, batch_input):

		batch_audio_output = self.model(batch_input).last_hidden_state[:,0,:] # (batch, 768)
		audio_logit = self.W(batch_audio_output) # (batch, clsNum)
		
		return batch_audio_output, audio_logit


class Student_Video(nn.Module):
	def __init__(self, video_model, clsNum):
		super(Student_Video, self).__init__()
		
		vmodel_path = video_model
		if video_model == "facebook/timesformer-base-finetuned-k400":

			self.model = TimesformerModel.from_pretrained(vmodel_path)

		self.hiddenDim = self.model.config.hidden_size
			
		"""score"""
		self.W = nn.Linear(self.hiddenDim, clsNum)

	def forward(self, batch_input):

		batch_video_output = self.model(batch_input).last_hidden_state[:,0,:] # (batch, 768)
		video_logit = self.W(batch_video_output) # (batch, clsNum)
		
		return batch_video_output, video_logit


class ASFTV(nn.Module):
	"""text and vision modalities only"""
	def __init__(self, clsNum, hidden_size, beta_shift, dropout_prob, num_head):
		super().__init__()
		self.TEXT_DIM = 768
		self.VISUAL_DIM = 768
		self.multihead_attn = nn.MultiheadAttention(self.VISUAL_DIM, num_head)

		self.W_hav = nn.Linear(self.VISUAL_DIM + self.TEXT_DIM, self.TEXT_DIM)

		self.W_av = nn.Linear(self.VISUAL_DIM, self.TEXT_DIM)

		self.beta_shift = beta_shift

		self.LayerNorm = nn.LayerNorm(hidden_size)
		self.AV_LayerNorm = nn.LayerNorm(self.VISUAL_DIM)
		self.dropout = nn.Dropout(dropout_prob)
		
		"""Logit"""
		self.W = nn.Linear(self.TEXT_DIM, clsNum)
	
	def forward(self, text_embedding, visual):
		eps = 1e-6
		# nv_embedd = torch.cat((visual, acoustic), dim=-1)
		new_nv = self.multihead_attn(visual, visual, visual)[0] + visual
		
		av_embedd = self.dropout(self.AV_LayerNorm(new_nv))

		weight_av = F.relu(self.W_hav(torch.cat((av_embedd, text_embedding), dim=-1)))

		h_m = weight_av * self.W_av(av_embedd)

		em_norm = text_embedding.norm(2, dim=-1)
		hm_norm = h_m.norm(2, dim=-1)

		hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).cuda()
		hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

		thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

		ones = torch.ones(thresh_hold.shape, requires_grad=True).cuda()

		alpha = torch.min(thresh_hold, ones)
		alpha = alpha.unsqueeze(dim=-1)

		acoustic_vis_embedding = alpha * h_m

		embedding_output = self.dropout(
			self.LayerNorm(acoustic_vis_embedding + text_embedding)
		)
		
		logits = self.W(embedding_output)
		return logits


def get_models():
	'''teacher model load'''
	cur_dir = osp.abspath(osp.dirname(__file__))
	print(f'cur dir: {cur_dir} \n****')
	save_model_dir = osp.join(cur_dir, '../../TelME/MELD/MELD/save_model')
	text_model = "roberta-large"
	video_model = "facebook/timesformer-base-finetuned-k400"
	clsNum = 7
	model_t = Teacher_model(text_model, clsNum)
	# model_t.load_state_dict(torch.load('./MELD/save_model/teacher.bin'))
	model_t.load_state_dict(torch.load(osp.join(save_model_dir, 'teacher.bin')))
	model_t.requires_grad_(False)
	model_t = model_t.cuda()
	model_t.eval()

	video_s = Student_Video(video_model, clsNum)
	# video_s.load_state_dict(torch.load('./MELD/save_model/student_video/total_student.bin')) 
	video_s.load_state_dict(torch.load(osp.join(save_model_dir, 'student_video/total_student.bin')))
	video_s.requires_grad_(False)
	video_s = video_s.cuda()
	video_s.eval()

	'''fusion'''
	hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.2, 3
	fusion = ASFTV(clsNum, hidden_size, beta_shift, dropout_prob, num_head)  
	# fusion.load_state_dict(torch.load('./MELD/save_model/total_fusion_tv.bin'))
	fusion.load_state_dict(torch.load(osp.join(save_model_dir, 'total_fusion_tv.bin')))
	fusion.requires_grad_(False)
	fusion = fusion.cuda()
	fusion.eval()
	return model_t, video_s, fusion


model_t, model_s, fusion_module = get_models()
	
async def telme_inference(input_tokens, attention_masks, video_inputs):
	text_hidden, test_logits = model_t(input_tokens, attention_masks)
	video_hidden, video_logits = model_s(video_inputs)  #  torch.Size([4, 8, 3, 224, 224])
	logits = fusion_module(text_hidden, video_hidden)
	# print(f'get emotion response logits: {logits} \n =================')
	emotion_label = torch.argmax(logits, dim=-1).item()
	
	if emotion_label == Emotions.Neutral:
		anim = await emo_recognizer.on_emotion_recog_task(frame_buffer.buffer_content[-1])
		print(f'anim from gpt: {anim} \n*****')
		return anim
	anim = random.choice(EMOTION_TO_ANIM.get(emotion_label, []))
	return anim



async def get_emotion_response(ts_end, duration):
	input_tokens, attention_masks = get_text_inputs_from_raw_telme()
	# print(f'type of text input ids: {type(text_input_ids)}, {len(text_input_ids)}, {text_input_ids.shape} \n *****')
	# text_input_ids = torch.tensor(diag_buffer.get_text_inputs_ids())
	video_inputs = get_vision_inputs_from_raw_telme(ts_end, duration)
	# print(f'img inputs: {img_inputs.shape}, img_mask :{img_mask.shape} \n &&&&&&&&&&&')
	# img_inputs, img_mask = get_img_inputs_from_raw()
	# reps, logits = model(text_input_ids.unsqueeze(0).cuda(), img_inputs.unsqueeze(0).cuda(), img_mask.unsqueeze(0).cuda())

	anim = await telme_inference(input_tokens.cuda(), attention_masks.cuda(), video_inputs.cuda())
	return anim


if __name__ == "__main__":
	cur_dir = osp.abspath(osp.dirname(__file__))
	print(f'cur dir: {cur_dir} \n****')


