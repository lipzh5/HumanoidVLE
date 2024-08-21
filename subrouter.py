# -*- coding:utf-8 -*-
# @Author: Peizhen Li 
# @Desc: None
import sys
import os
import zmq
import zmq.asyncio
from zmq.asyncio import Context
from CONF import *
from utils import frame_buffer
from const import ResponseCode
from models import vle_model




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

	async def sub_vcap_data(self):
		while True:
			try:
				data = await self.vcap_sub_sock.recv()
				frame_buffer.append_content(data)
				if len(frame_buffer) % 100 == 0:
					print(f'frame buffer len: {len(frame_buffer)}')
			except Exception as e:
				print(str(e))
				import traceback
				traceback.print_stack()
				print(f'===============')

		
	
	async def route_vle_task(self):
		try:
			while True:
				msg = await self.text_router_sock.recv_multipart()
				identity = msg[0]
				print(f'msg len:{len(msg)}')
				print('route visual task identity: ', identity)
				try:
					# res_code, ans = await self.deal_visual_task(*msg[1:])
					# if ans is None:
					# 	ans = 'None'
					res_code = ResponseCode.Success
					ans = "None"
					print(f'task answer:{ans} \n ------- ')
					resp = [identity, res_code]
					if isinstance(ans, list) or isinstance(ans, tuple):
						resp.extend([item.encode(encoding) for item in ans])
					else:
						resp.append(ans.encode(encoding))
				except Exception as e:
					print(str(e))
					print(f'msg: {msg}')
					print('----------')
					resp = [identity, ResponseCode.Fail, b'None']

				await self.text_router_sock.send_multipart(resp)
		except Exception as e:
			print(str(e))
			print(f'-----------route vle task error---------')
			import traceback
			traceback.print_stack()