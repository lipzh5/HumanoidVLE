# -*- coding:utf-8 -*-
# @Author: Peizhen Li 
# @Desc: None
import sys
import os
import zmq
import zmq.asyncio
from zmq.asyncio import Context
from CONF import *
from const import *
from data_buffers import frame_buffer, diag_buffer
from const import ResponseCode
from models import vle_model
# from models import telme_model
import time
import io
from io import BytesIO
from PIL import Image



ctx = Context.instance()

class SubRouter:
	def __init__(self):
		super().__init__()
		self.vcap_sub_sock = ctx.socket(zmq.SUB)
		self.vcap_sub_sock.setsockopt(zmq.SUBSCRIBE, b'')
		self.vcap_sub_sock.setsockopt(zmq.CONFLATE, 1)
		self.vcap_sub_sock.connect(vsub_addr)

		self.text_router_sock = ctx.socket(zmq.ROUTER)
		self.text_router_sock.connect(trout_addr)
		self.ameca_just_said  = False
		self.ameca_sentence = b''   # Ameca may send sentence in several pieces

	async def sub_vcap_data(self):
		while True:
			try:
				data = await self.vcap_sub_sock.recv()
				frame_buffer.append_content(data)
				if len(frame_buffer) % 100 == 0:
					print(f'frame buffer len: {len(frame_buffer)}')
				# print(f'time: {time.time()}')
			except Exception as e:
				print(str(e))
				import traceback
				traceback.print_stack()
				print(f'===============')
	
	async def get_emotion_response(self, utterance, ts_end, duration, from_ameca):
		print(f'ts_end :{ts_end}, duration: {duration} \n***** ')
		if utterance == b'ameca starts new dialogue':
			diag_buffer.clear_buffer()
			print(f'starts new dialogue!!!\n*****')
			return ResponseCode.KeepSilent, ''
		
		if int(from_ameca.decode(encoding)) > 0:  # sentence from Ameca
			self.ameca_just_said = True
			self.ameca_sentence += (b' ' + utterance)
			return ResponseCode.KeepSilent, ''
		else: 
			if self.ameca_just_said:
				print(f'ameca sentence: {self.ameca_sentence} \n*****')
				self.ameca_just_said = False   # human starts to respond to ameca
				self.ameca_sentence and diag_buffer.update_dialogue(self.ameca_sentence)
				self.ameca_sentence = b''       # add cached ameca sentence as well as current human dialogue
				diag_buffer.update_dialogue(utterance) 
			else:
				diag_buffer.update_dialogue(utterance)
		
		# print(f'frame buffer arrival time: {frame_buffer.arrival_time} \n *****')
		# img = Image.open(BytesIO(frame_buffer.buffer_content[-1]))
		# img.save(f'./assets/1/debug_{time.time()}.png')   # TODO 
		# NOTE: comparison with TelME
		emo_anim = await vle_model.get_emotion_response(float(ts_end.decode(encoding)), float(duration.decode(encoding)))
		
		'''debug save the last frame'''
		# emo_anim = await telme_model.get_emotion_response(float(ts_end.decode(encoding)), float(duration.decode(encoding)))
		# print(f'emotion emo_anim : {emo_anim }')
		return ResponseCode.Success, emo_anim 
		
		
	
	async def route_vle_task(self):
		try:
			while True:
				msg = await self.text_router_sock.recv_multipart()
				identity = msg[0]
				# print(f'msg len:{len(msg)}')
				# print('route visual task identity: ', identity)
				task_type = msg[1]
				# print(f'task type: {task_type} \n ******')
				try:
					
					res_code, ans = await self.get_emotion_response(*msg[2:])
					if ans is None:
						ans = 'None'
					res_code = ResponseCode.Success
					print(f'task answer:{ans} \n ------- ')
					resp = [identity, res_code]
					if isinstance(ans, list) or isinstance(ans, tuple):
						resp.extend([item.encode(encoding) for item in ans])
					else:
						resp.append(ans.encode(encoding))
				except Exception as e:
					print('===========exception=======')
					print(str(e))
					print('===========exception=======')
					print(f'msg: {msg}')
					print('----------')
					resp = [identity, ResponseCode.Fail, b'None']

				await self.text_router_sock.send_multipart(resp)
		except Exception as e:
			print(str(e))
			print(f'-----------route vle task error---------')
			import traceback
			traceback.print_stack()