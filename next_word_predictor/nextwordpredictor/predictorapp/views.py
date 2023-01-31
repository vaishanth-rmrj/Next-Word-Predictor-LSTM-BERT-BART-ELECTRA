from django.shortcuts import render_to_response
from rest_framework.decorators import api_view
from rest_framework.response import Response
# Create your views here.
import os
from pickle import load
from predictormodel.model import LSTMNextWordModel, BERTNextWordModel, BARTNextWordModel
# from keras import backend as K



# ENGLISH_MODEL_PATH = BASE_DIR + '/predictormodel/models/dage-english-2020-model.h5'
# ENGLISH_TOKENIZER_PATH = BASE_DIR + '/predictormodel/tokenizers/dage-english-2020-tokenizer.pkl'

# CHINESE_MODEL_PATH = BASE_DIR + '/predictormodel/models/chinese-20-model.h5'
# CHINESE_TOKENIZER_PATH = BASE_DIR + '/predictormodel/tokenizers/chinese-20-tokenizer.pkl'

# MALAY_MODEL_PATH = BASE_DIR + '/predictormodel/models/dage-malay-model.h5'
# MALAY_TOKENIZER_PATH = BASE_DIR + '/predictormodel/tokenizers/dage-malay-tokenizer.pkl'

# MALAYSIAN_MODEL_PATH = BASE_DIR + '/predictormodel/models/malaysian-20-model.h5'
# MALAYSIAN_TOKENIZER_PATH = BASE_DIR + '/predictormodel/tokenizers/malaysian-20-tokenizer.pkl'

current_mode = 'EN'
print('mode:', current_mode)

# def load_language(l, tokenizer, m, overwrite_model=False):
# 	if l == 'EN':
# 		tokenizer_path = ENGLISH_TOKENIZER_PATH
# 		model_path = ENGLISH_MODEL_PATH
# 	elif l == 'CN':
# 		tokenizer_path = CHINESE_TOKENIZER_PATH
# 		model_path = CHINESE_MODEL_PATH
# 	elif l == 'MY':
# 		tokenizer_path = MALAYSIAN_TOKENIZER_PATH
# 		model_path = MALAYSIAN_MODEL_PATH
# 	else:
# 		tokenizer_path = MALAY_TOKENIZER_PATH
# 		model_path = MALAY_MODEL_PATH
# 	print('Current language is', l)
# 	print('Loading tokenizer...')
# 	tokenizer = load(open(tokenizer_path, 'rb'))
# 	tokenizer.oov_token = None
# 	print('\tDONE.')
# 	if overwrite_model:
# 		del m
# 		print('Deleted model.')
# 		K.clear_session()
# 		print('Cleared backend session.')
# 		global model
# 		model = NextWordModel()
# 		model.load_model_and_tokenizer(tokenizer, model_path)

# print('Loading tokenizer...')
# tokenizer = load(open(ENGLISH_TOKENIZER_PATH, 'rb'))
# tokenizer.oov_token = None
# print('\tDONE.')

# model = NextWordModel()
# model.load_model_and_tokenizer(tokenizer, ENGLISH_MODEL_PATH)

# seq_length = {'EN': 19, 'CN': 19, 'MY': 19, 'ML': 20}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LSTM_MODEL_PATH = BASE_DIR + '/predictormodel/models/lstm_model.h5'
# LSTM_TOKENIZER_PATH = BASE_DIR + '/predictormodel/tokenizers/lstm_tokenizer.pkl'

# lstm_model = LSTMNextWordModel(LSTM_MODEL_PATH, LSTM_TOKENIZER_PATH)

BERT_MODEL_DIR_PATH = BASE_DIR + '/predictormodel/models/bert/'
bert_model = BERTNextWordModel(BERT_MODEL_DIR_PATH)

BART_MODEL_DIR_PATH = BASE_DIR + '/predictormodel/models/bart/'
bart_model = BARTNextWordModel(BART_MODEL_DIR_PATH)

@api_view(['GET', 'POST'])
def predict(request):
	if request.method == 'POST':
		pretext = request.data['text']
		# lstm_preds = lstm_model.predict_next_word(pretext)
		bert_preds = bert_model.predict_next_word(pretext)
		bart_preds = bart_model.predict_next_word(pretext)
		# print(pretext)
		# print(bart_preds)
# 		if current_mode == 'CN':
# 			pretext = ' '.join([t for t in pretext])
# 		predicted_words = model.generate_seq(seq_length[current_mode], pretext, 1)
		return Response({
			"lstm_predictions": ["to", "there", "morning"],
			"bert_predictions": bert_preds,
			"bart_predictions": bart_preds,
			})
	return Response({"text": "<TEXT HERE>"})

# @api_view(['GET', 'POST'])
# def change_mode(request):
# 	global current_mode, model, tokenizer
# 	if request.method == 'POST':
# 		current_mode = request.data['mode']
# 		load_language(current_mode, tokenizer, model, True)
# 		return Response({"changed_mode": current_mode})
# 	return Response({"current_mode": current_mode})

def index(request):
	return render_to_response('index.html')