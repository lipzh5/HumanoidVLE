# -*- coding:utf-8 -*-
# @Author: Peizhen Li 
# @Desc: None

from collections import deque
from CONF import frame_buffer_max_len
from transformers import AutoTokenizer


class FrameBuffer:
	_instance = None
	buffer_content = deque()

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
# CONTEXT_CONF['CLS'] = CLS  
# CONTEXT_CONF['SEP'] = SEP
# CONTEXT_CONF['mask_value'] = MASK


def pad_to_len(input_ids, max_len, pad_value):
	pass


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

	@classmethod
	def reset(cls):
		cls.dialogue = []

	@classmethod
	def get_text_inputs_ids(cls):
		query = 'For utterance:'
		query_ids = tokenizer(query)['input_ids'][1:-1]

		utterance_ids = []
		for idx, utt in enumerate(cls.dialogue):
			token_ids = tokenizer(text_with_speaker)['input_ids'][1:]
			utterance_ids.append(token_ids)
			full_context = [CLS]
			lidx = 0
			for lidx in range(idx):
				total_len = sum([len(item) for item in utterance_ids[lidx:]]) + 8
				if total_len + len(utterance_ids[idx]) <= max_len: # CONFIG['max_len']:
					break
			lidx = max(lidx, idx-8)
			for item in utterance_ids[lidx:]:
				full_context.extend(item)

			query_idx = idx
			# prompt = dialogue[query_idx]['speaker'] + ' feels <mask>'
			prompt = 'speaker feals <mask>'
			full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:]
			input_ids = full_context + full_query
			input_ids, _ = pad_to_len(input_ids, max_len, pad_value) # CONFIG['max_len'], CONFIG['pad_value']
			# ret_utterances.append(input_ids)
			# ret_labels.append(dialogue[query_idx]['label'])

	





	dialogues = load_meld_turn(anno_csv_dir, split_type, vocab_path)

	ret_utterances = []
	ret_labels = []
	tokenizer = get_tokenizer(pretrained_path)  # len: 50265
	for dialogue in dialogues:
		utterance_ids = []
		query = 'For utterance:'
		query_ids = tokenizer(query)['input_ids'][1:-1]
		for idx, turn_data in enumerate(dialogue):
			text_with_speaker = turn_data['speaker'] + ':' + turn_data['text']
			token_ids = tokenizer(text_with_speaker)['input_ids'][1:]
			utterance_ids.append(token_ids)
			if turn_data['label'] < 0:
				continue
			full_context = [CONFIG['CLS']]
			lidx = 0
			for lidx in range(idx):
				total_len = sum([len(item) for item in utterance_ids[lidx:]]) + 8
				if total_len + len(utterance_ids[idx]) <= max_len: # CONFIG['max_len']:
					break
			lidx = max(lidx, idx-8)
			for item in utterance_ids[lidx:]:
				full_context.extend(item)

			query_idx = idx
			prompt = dialogue[query_idx]['speaker'] + ' feels <mask>'
			full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:]
			input_ids = full_context + full_query
			input_ids, _ = pad_to_len(input_ids, max_len, pad_value) # CONFIG['max_len'], CONFIG['pad_value']
			ret_utterances.append(input_ids)
			ret_labels.append(dialogue[query_idx]['label'])

			# if train and idx > 3 and torch.rand(1).item() < 0.2:
			#     query_idx = random.randint(lidx, idx-1)
			#     if dialogue[query_idx]['label'] < 0:
			#         continue
			#     prompt = dialogue[query_idx]['speaker'] + ' feels <mask>'
			#     full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:]
			#     input_ids = full_context + full_query
			#     input_ids = pad_to_len(input_ids, max_len, pad_value) # CONFIG['max_len'], CONFIG['pad_value']
			#     ret_utterances.append(input_ids)
			#     ret_labels.append(dialogue[query_idx]['label'])
			
	# print(f'len ret utterance: {len(ret_utterances)}, {ret_utterances[0]}')

	self.text_input_ids = torch.tensor(ret_utterances, dtype=torch.long)  # additional training pairs
	self.labels = torch.tensor(ret_labels, dtype=torch.long)


frame_buffer = FrameBuffer()

