
ATTN_MASK_FILL = -1e38 # -1e-9  #

FPS = 25 

ORIGINAL_IMG_SHAPE = (720, 1280, 3)  # (h, w, c)
FACE_IMG_SHAPE = (160, 160)


class ResponseCode:
	KeepSilent = b'0' 
	Success = b'1'
	Fail = b'2'



class Emotions:
	Other = -1
	Neutral = 0
	Surprise = 1
	Fear = 2
	Sadness = 3
	Joy = 4
	Disgust = 5
	Anger = 6

	# Angry = 1
	# Confused = 2
	# Dislike = 3
	# Fear = 4
	# Happy = 5
	# Sad = 6
	# Scared = 7
	# Surprised = 8


EMOTION_TO_ANIM = {
	Emotions.Other: ['Chat Expressions.dir/Chat_G2_Neutral.project',],
	Emotions.Neutral: ['Chat Expressions.dir/Chat_G2_Neutral.project',],
	Emotions.Surprise: [
		'Chat Expressions.dir/Chat_G2_Surprised_1.project',
		'Chat Expressions.dir/Chat_G2_Surprised_2.project',],
	Emotions.Fear: [
		'Chat Expressions.dir/Chat_G2_Fear_1.project',
		'Chat Expressions.dir/Chat_G2_Fear_2.project',],
	
	Emotions.Sadness: [
		'Chat Expressions.dir/Chat_G2_Sad_1.project',
		'Chat Expressions.dir/Chat_G2_Sad_2.project',],
	
	Emotions.Joy: [
		'Chat Expressions.dir/Chat_G2_Happy_1.project',
		'Chat Expressions.dir/Chat_G2_Happy_2.project',
		'Chat Expressions.dir/Chat_G2_Happy_with_audio.project',],
	
	Emotions.Disgust: [
		'Chat Expressions.dir/Chat_G2_Dislike_1.project',
		'Chat Expressions.dir/Chat_G2_Dislike_2.project',],

	Emotions.Anger: [
		'Chat Expressions.dir/Chat_G2_Angry_1.project',
		'Chat Expressions.dir/Chat_G2_Angry_2.project',
		'Chat Expressions.dir/Chat_G2_Angry_3.project',],

	# Emotions.Confused: ['Chat Expressions.dir/Chat_G2_Confused_1.project',],
	
	# Emotions.Scared: ['Chat Expressions.dir/Chat_G2_Scared_1.project',],
	

}

