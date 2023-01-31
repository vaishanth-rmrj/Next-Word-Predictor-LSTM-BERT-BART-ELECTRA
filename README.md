# Next-Word-Predictor-LSTM-BERT-BART-ELECTRA
A next word predictor app implemented using LSTM, BERT. BART, ELECTRA with a Django Web Interface.

Objective: to predict what word comes next with Tensorflow and Pytorch. Implement LSTM, BERT, BART, ELECTRA.

## Project Intro
The purpose of this project is to train next word predicting models. Models should be able to suggest the next word after user has input word/words. Four models are used for the prediction. Python Django as backend and JavaScript/HTML as Frontend.

### Methods Used
* Language Prediction
* Natural Language Processing
* LSTM, BERT, BART, ELECTRA

### Technologies
* Python
* Python Django
* Tensorflow, Keras, Pytorch
* Js, HTML

## Project Description
* `model_predictions.ipynb` - Python interactive notebook used to train the model and predict next word.
* `nextwordpredictor` - Django application, loads trained models

Data are from following sources:
1. English - [blogger.com](http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm)

## Process Flow
- data collection
- data processing/cleaning
- words tokenizing
- model training
- frontend development

### Prerequisites
Install python dependencies via command
`pip install -r requirements.txt`

1. Start server via command `python3 manage.py runserver` from the nextwordpredictor/ directory and navigate to `http://127.0.0.1:8000/index`.

<img src="https://github.com/vaishanth-rmrj/Next-Word-Predictor-LSTM-BERT-BART-ELECTRA/blob/main/extras/next_word_pred_app_gui.png" width=800/>



