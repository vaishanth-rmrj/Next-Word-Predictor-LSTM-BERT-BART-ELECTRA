import numpy as np
import pickle
import string

import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertForMaskedLM
from tensorflow.keras.models import load_model

class LSTMNextWordModel:
    def __init__(self, lstm_model_path, lstm_tokenizer_path):
        # laoding model and tokenizer
        self.model = load_model(lstm_model_path)
        self.tokenizer = pickle.load(open(lstm_tokenizer_path, 'rb'))
    
    def predict_next_word(self, sentence_text):
        # split the last word from sentence
        last_word = sentence_text.split(" ")[-1]

        # convert text to tokens
        encoded_txt = self.tokenizer.texts_to_sequences([last_word])[0]
        encoded_txt = np.array(encoded_txt)

        # predict the next word
        next_word_token_prob = self.model.predict(encoded_txt)[0]
        # select the 4 high prob words
        next_word_toekn_max_probs = np.argsort(next_word_token_prob, axis=0)[-4:]

        preds = []

        for pred_token in next_word_toekn_max_probs:
            for word, token in self.tokenizer.word_index.items():
                if token == pred_token:
                    # append pred word and its probability
                    # preds.append([word, next_word_token_prob[pred_token]])
                    # append just words
                    preds.append(word)
            
        return preds

class BERTNextWordModel:
    def __init__(self, model_dir_path):
        # laoding model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_dir_path)
        self.model = AutoModel.from_pretrained(model_dir_path,local_files_only=True).eval()

    def decode(self, pred_idx, top_clean):
        ignore_tokens = string.punctuation + '[PAD]'
        tokens = []
        for w in pred_idx:
            token = ''.join(self.tokenizer.decode(w).split())
            if token not in ignore_tokens:
                tokens.append(token.replace('##', ''))
        return '\n'.join(tokens[:top_clean])


    def encode(self, text_sentence, add_special_tokens=True):
        text_sentence = text_sentence.replace('<mask>', self.tokenizer.mask_token)
        # if <mask> is the last token, append a "." so that models dont predict punctuation.
        if self.tokenizer.mask_token == text_sentence.split()[-1]:
            text_sentence += ' .'

        input_ids = torch.tensor([self.tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
        mask_idx = torch.where(input_ids == self.tokenizer.mask_token_id)[1].tolist()[0]
        return input_ids, mask_idx

    def predict_next_word(self, sentence_txt, num_words=4, top_clean=5):

        # encode text
        print(sentence_txt)
        input_ids, mask_idx = self.encode(sentence_txt)

        # predict next words bert
        with torch.no_grad():
            predict = self.model(input_ids)[0]

        next_words = self.decode(predict[0, mask_idx, :].topk(num_words).indices.tolist(), top_clean)
        next_words = next_words.split("\n")
        return (next_words)

class BARTNextWordModel:
    def __init__(self, model_dir_path):
        # laoding model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir_path)
        self.model = AutoModel.from_pretrained(model_dir_path, local_files_only=True).eval()

    def decode(self, pred_idx, top_clean):
        ignore_tokens = string.punctuation + '[PAD]'
        tokens = []
        for w in pred_idx:
            token = ''.join(self.tokenizer.decode(w).split())
            if token not in ignore_tokens:
                tokens.append(token.replace('##', ''))
        return '\n'.join(tokens[:top_clean])


    def encode(self, text_sentence, add_special_tokens=True):
        text_sentence = text_sentence.replace('<mask>', self.tokenizer.mask_token)
        # if <mask> is the last token, append a "." so that models dont predict punctuation.
        if self.tokenizer.mask_token == text_sentence.split()[-1]:
            text_sentence += ' .'

        input_ids = torch.tensor([self.tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
        mask_idx = torch.where(input_ids == self.tokenizer.mask_token_id)[1].tolist()[0]
        return input_ids, mask_idx

    def predict_next_word(self, sentence_txt, num_words=4, top_clean=5):

        # encode text
        print(sentence_txt)
        input_ids, mask_idx = self.encode(sentence_txt)

        # predict next words bart
        with torch.no_grad():
            predict = self.model(input_ids)[0]

        next_words = self.decode(predict[0, mask_idx, :].topk(num_words).indices.tolist(), top_clean)
        next_words = next_words.split("\n")
        return (next_words)