# -*- coding:utf-8 -*-
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

face_detector = MTCNN(keep_all=True, post_process=False, select_largest=False)


def get_center_faces(img_arr):
	boxes, probs = face_detector.detect(img_arr)    # boxes: Nx4 array
	box_order = np.argsort(np.abs((boxes[:, 2] + boxes[:, 0]) /2 - 640.))  # [::-1]
	selected_boxes = boxes[0].reshape(-1, 4)
	faces = face_detector.extract(img_arr, selected_boxes, save_path=None)
	# faces = face_detector.extract(img_arr, selected_boxes, save_path='./assets/center_face.png')
	return faces

def test_mtcnn():
	import torch
	import cv2
	import os.path as osp
	cur_dir = osp.abspath(osp.dirname(__file__))
	print(f'cur dir: {cur_dir} \n *****')
	img_arr = np.asarray(Image.open(osp.join(cur_dir, 'assets/multipersons.jpg')))
	img_arr = cv2.resize(img_arr, dsize=(1280, 720), interpolation=cv2.INTER_AREA)  # (720, 1280, 3)
	print(f'img arr shape: {img_arr.shape}')
	# cv2.imwrite('./assets/resized_multi.png', img_arr)
	# raise ValueError('Penny stops here!!!')
	face_tensors = get_center_faces(img_arr)
	# debug_face = face_tensors[0].permute(1, 2, 0).numpy()
	# cv2.imwrite('./assets/debug_save_face.png', cv2.cvtColor(debug_face, cv2.COLOR_RGB2BGR))
	# print(f'face tesnors0 : {torch.sum(face_tensors[0])}, torch.max: {torch.max(face_tensors[0])}')
	# print(f'face tensors[0]: {face_tensors[0].shape}, {torch.permute(face_tensors[0], (1, 2, 0)).shape}')
	# print(f'face tensor 0: {face_tensors[0].permute(1,2,0).numpy().shape}')
	# cv2.imwrite('./assets/debug_save_face.png', face_tensors[0].permute(1,2,0).numpy())

	cv2.imwrite('./assets/debug_save_face.png', face_tensors[0].permute(1,2,0).numpy())

	# print(f'shape face tensors: {face_tensors.shape}')  # torch.Size([1, 3, 160, 160])




async def test_emotion_response():
	from models import vle_model
	from utils import diag_buffer
	diag_buffer.update_dialogue(b'enen')
	# diag_buffer.update_dialogue(b'how are you? ')
	# diag_buffer.update_dialogue(b'nice to meet you!')
	# diag_buffer.update_dialogue(b'I really hate you!')
	# diag_buffer.update_dialogue(b'Could you tell me your name?')
	# diag_buffer.update_dialogue(b'I do not understand what you meant')
	anim = await vle_model.get_emotion_response(0, 0)
	print(anim)


def test_vision_input():
	from models.vle_model import model
	from data_buffers import diag_buffer
	from utils import get_text_inputs_from_raw, pad_to_len, normalize, transform, resize
	import os.path as osp
	from PIL import Image
	import numpy as np
	import torch
	import random
	from const import EMOTION_TO_ANIM
	# diag_buffer.update_dialogue(b'Someone one the subway licked my neck! Licked my neck!')
	# diag_buffer.update_dialogue(b'On Willies still alive!')
	# diag_buffer.update_dialogue(b'What are you guys doing?')
	# diag_buffer.update_dialogue(b'On, my mom called, theyre gonna run our engagement announcement in the local paper, so we were looking for a good picture of us.')
	# diag_buffer.update_dialogue(b'Oooh, I am afraid that does not exist.')
	diag_buffer.update_dialogue(b'i am still working on my experiments')
	# diag_buffer.update_dialogue(b'What?')
	text_input_ids = get_text_inputs_from_raw()
	# debug_face_dir = './assets/dia4_utt1'
	# debug_face_dir = './assets/debug_faces_sad'
	debug_face_dir = './assets/debug_faces_happy'
	all_faces = []
	ref_frame = None
	# for _ in range(5):
	for i in range(1, 20):
		# face_path = osp.join(debug_face_dir, f'frame_det_00_{str(i).zfill(6)}.bmp')
		face_path = osp.join(debug_face_dir, f'debug_face_{i}.png')
		img_arr = np.asarray(Image.open(face_path))
		if ref_frame is None:
			ref_frame = img_arr
		img_arr = img_arr - ref_frame
		all_faces.append(transform(resize(img_arr)))

	all_faces = torch.stack(all_faces)
	# all_faces = torch.zeros([100, 3, 160, 160])
	# print(f'max faces: {}')
	# all_faces = torch.rand([100, 3, 160, 160])
	
	print(f'all faces.shape: {all_faces.shape}')
	all_faces, mask = pad_to_len(all_faces, 100, pad_value=0)
	print(f'all faces22: {all_faces.shape}, mask shape:{mask.shape}')
	logits = model(text_input_ids.unsqueeze(0).cuda(), all_faces.unsqueeze(0).cuda(), mask.unsqueeze(0).cuda())
	print(f'logits: {logits} \n****')
	emotion_label = torch.argmax(logits, dim=-1).item()
	anim = random.choice(EMOTION_TO_ANIM.get(emotion_label, []))
	print(f'anim: {anim}')


def test_emo_rec():
	from models.emotion_rec import emo_recognizer
	import asyncio
	asyncio.run(emo_recognizer.on_multimodal_emotion_recog_task(b'', 'one'))
	# asyncio.run(emo_recognizer.on_emotion_recog_task(b''))
	pass

# def dummy_args(*args):
# 	print(args)
# 	time, duration = args
# 	print(f'time: {time}, duration: {duration} \n****')
	




if __name__ == "__main__":
	asyncio.run(run_sub_router())
	# context = 'i am happy today'
	# prompt = f"""You are talking with the person in front of you and guess the person's emotion base on the conversation context: {context}
	# 		and observation in the form of image, candidate_emotions are: -1.not provided,
	# 		 -1.other, 0.neutral, 1.surprise, 2.fear, 3.sadness, 4.joy, 5.disgust, 6.anger.
	# 		 you should provide the emotion only, e.g., 1.surprise.
	# 		 """
	# print(f'prompt: {prompt}')
	# asyncio.run(test_emotion_response())
	# test_emo_rec()
	# test_vision_input()
	# test_emotion_response()
	# test_mtcnn()