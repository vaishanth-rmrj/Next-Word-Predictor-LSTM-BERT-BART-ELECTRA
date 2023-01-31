# Importing the Libraries
import torch
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import string

def lstm_next_word_pred(model, tokenizer, sentence):
  # split the last word from sentence
  last_word = sentence.split(" ")[-1]

  # convert text to tokens
  encoded_txt = tokenizer.texts_to_sequences([last_word])[0]
  encoded_txt = np.array(encoded_txt)

  # predict the next word
  next_word_token_prob = model.predict(encoded_txt)[0]
  # select the 4 high prob words
  next_word_toekn_max_probs = np.argsort(next_word_token_prob, axis=0)[-4:]

  preds = []

  for pred_token in next_word_toekn_max_probs:
    for word, token in tokenizer.word_index.items():
      if token == pred_token:
        # append pred word and its probability
        preds.append([word, next_word_token_prob[pred_token]])
  
  return preds

def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx

####  using pre trained models to predict words ####

def bert_next_word_pred(bert_model, bert_tokenizer, text_sentence, num_words, top_clean=5):

  # encode text
  input_ids, mask_idx = encode(bert_tokenizer, text_sentence)

  # predict next words bert
  with torch.no_grad():
      predict = bert_model(input_ids)[0]

  next_words = decode(bert_tokenizer, predict[0, mask_idx, :].topk(num_words).indices.tolist(), top_clean)
  
  return (next_words)

def bart_next_word_pred(bart_model, bart_tokenizer, text_sentence, num_words, top_clean=5):
  
  # encode text
  input_ids, mask_idx = encode(bart_tokenizer, text_sentence, add_special_tokens=True)
  
  # predict next words using bart
  with torch.no_grad():
      predict = bart_model(input_ids)[0]

  next_words = decode(bart_tokenizer, predict[0, mask_idx, :].topk(num_words).indices.tolist(), top_clean)

  return (next_words)


def electra_next_word_pred(electra_model, electra_tokenizer, text_sentence, num_words, top_clean=5):
  # encode text
  input_ids, mask_idx = encode(electra_tokenizer, text_sentence, add_special_tokens=True)
  
  # predict next words using electra
  with torch.no_grad():
      predict = electra_model(input_ids)[0]

  next_words = decode(electra_tokenizer, predict[0, mask_idx, :].topk(num_words).indices.tolist(), top_clean)

  return (next_words)

























