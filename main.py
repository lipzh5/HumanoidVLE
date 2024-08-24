# -*- coding:utf-8 -*-
# @Author: Peizhen Li 
# @Desc: None
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import asyncio
from subrouter import SubRouter


async def run_sub_router():
	sub_router = SubRouter()
	loop = asyncio.get_event_loop()
	print(f'task loop is running!!!')
	task1 = loop.create_task(sub_router.sub_vcap_data())
	task2 = loop.create_task(sub_router.route_vle_task())
	await asyncio.gather(task1, task2)


from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
face_detector = MTCNN(keep_all=True)
def test_mtcnn():
	img_arr = np.asarray(Image.open('debug.png'))
	boxes, probs = face_detector.detect(img_arr)
	print(f'type boxes: {type(boxes)}, {boxes.shape}\n {boxes}')
	face_tensors = face_detector(img_arr, save_path='debug_face.png')
	# face = mtcnn.preprocess(face_img, bbox_, points_, image_size=args.face_size)

	print(f'shape face tensors: {face_tensors.shape}')



def test_emotion_response():
	from models import vle_model
	from utils import diag_buffer
	diag_buffer.update_dialogue(b'how are you? ')
	diag_buffer.update_dialogue(b'nice to meet you!')
	diag_buffer.update_dialogue(b'I really hate you!')
	diag_buffer.update_dialogue(b'Could you tell me your name?')
	diag_buffer.update_dialogue(b'I do not understand what you meant')
	anim = vle_model.get_emotion_response(0, 0)
	print(anim)

	pass

if __name__ == "__main__":
	# test_emotion_response()
	# test_mtcnn()
	asyncio.run(run_sub_router())
   