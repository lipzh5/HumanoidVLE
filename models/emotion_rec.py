# -*- coding:utf-8 -*-
# @Desc: Emotion recognition using gpt-4o 

import aiohttp
import asyncio
import os
import time
import random
import base64
import requests
from PIL import Image
import io
from io import BytesIO
from const import EMOTION_TO_ANIM
from utils import api_key

def encode_image(image_path):
	with open(image_path, 'rb') as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')


class EmotionRecognizer:
    def __init__(self):
        self.headers = {
		"Content-Type": "application/json",
		"Authorization": f"Bearer {api_key}"}
        self.payload = {
		"model": "gpt-4o", 
		"messages": [
					{
						"role": "user",
					}
				],
		"max_tokens": 250,}
    
    # # TODO debug only
    # def get_center_faces(self, save_path=None):
    #     from facenet_pytorch import MTCNN
    #     from const import ORIGINAL_IMG_SHAPE
    #     face_detector = MTCNN(keep_all=True, post_process=False, select_largest=False)
    #     '''ref to: https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py'''
    #     import numpy as np
    #     from PIL import Image
    #     """extract faces from raw image"""
    #     img_arr = np.asarray(Image.open('/home/penny/pycharmprojects/amecavle/assets/multipersons.jpg'))
    #     boxes, probs = face_detector.detect(img_arr)    # boxes: Nx4 array
    #     if boxes is None:
    #         return None
    #     box_order = np.argsort(np.abs((boxes[:, 2] + boxes[:, 0]) /2 - ORIGINAL_IMG_SHAPE[1]//2))  # [::-1]
    #     selected_boxes = boxes[1].reshape(-1, 4)
    #     faces = face_detector.extract(img_arr, selected_boxes, save_path='debug_multi_1.png')
    #     return faces



    async def on_emotion_recog_task(self, frame:bytes):
        img = Image.open(BytesIO(frame))
        # img.save('debug.png')   # TODO 
        b = io.BytesIO()
        # print(f'type of bytes io: {type(b)} \n *******')
        img.save(b, 'png')
        base64_image = base64.b64encode(b.getvalue()).decode('utf-8')
        # self.get_center_faces()
        # base64_image = encode_image('/home/penny/pycharmprojects/amecavle/assets/debug_multi.png') # neutral
        # base64_image = encode_image('/home/penny/pycharmprojects/amecavle/assets/debug_multi_1.png') # surprise
        content = [
            {"type": "text", 
            "text": """You are talking with the person in front of you and guess the person's emotion base on the observation in the form of image, candidate_emotions are: -1.not provided,
             -1.other, 0.neutral, 1.surprise, 2.fear, 3.sadness, 4.joy, 5.disgust, 6.anger.
             you should provide the emotion only, e.g., 1.surprise.
             """},
            {"type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }},
        ]
        self.payload['messages'][0]['content'] = content
        async with aiohttp.ClientSession() as session:
            response = await session.post(url="https://api.openai.com/v1/chat/completions",
                                        headers=self.headers,
                                        json=self.payload)
            res = await response.json()
            # print(f'response:: {res} \n*****')
            msg = res['choices'][0]['message']['content']
            print(msg)
            emo_label = int(msg.split('.')[0])
            emo_anim_lst = EMOTION_TO_ANIM.get(emo_label, [])
            print(f'emo list: {emo_anim_lst} \n****')
            if not emo_anim_lst:
                return ''
            return random.choice(emo_anim_lst)


# ===== example msg ======
''' -1.not provided

Based on the provided image, I cannot see your face or any significant facial features that would allow me to observe and identify your emotions.
type  spts: <class 'list'>, 
 ['-1.not provided', '', 'Based on the provided image, I cannot see your face or any significant facial features that would allow me to observe and identify your emotions.']
'''
# ==========================

emo_recognizer = EmotionRecognizer()


if __name__ == "__main__":
    emo_recog = EmotionRecognizer()
    asyncio.run(emo_recog.on_emotion_recog_task('b'))
    pass
