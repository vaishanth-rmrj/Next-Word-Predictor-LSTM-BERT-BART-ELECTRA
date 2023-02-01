# Next-Word-Predictor-LSTM-BERT-BART-ELECTRA
A next word predictor app implemented using LSTM, BERT, BART, ELECTRA with a Django Web Interface.

## Objective
The purpose of this project is to predict what word comes next with Tensorflow and Pytorch. Implement LSTM, BERT, BART, ELECTRA.

## Project Introduction
The Next-Word-Predictor-LSTM-BERT-BART-ELECTRA project aims to train next word predicting models. The models are able to suggest the next word after user inputs word/words. Four models are used for the prediction, including LSTM, BERT, BART, and ELECTRA. The backend is built with Python Django, while the frontend uses JavaScript and HTML.

## Methods Used
- Language Prediction
- Natural Language Processing
- LSTM, BERT, BART, ELECTRA

## Technologies
- Python
- Python Django
- Tensorflow, Keras, Pytorch
- JavaScript, HTML

## Project Description
- `model_predictions.ipynb`: a Python interactive notebook used to train the model and predict the next word.
- `nextwordpredictor`: a Django application that loads the trained models.

Data is collected from `blogger.com` and is processed, cleaned, and tokenized. The process flow includes the following steps:
1. Data collection: The first step is to collect data for the models. In this case, the data is collected from blogger.com.
2. Data processing and cleaning: Once the data is collected, it needs to be processed and cleaned to remove any irrelevant information and make it suitable for use in the models.
3. Word tokenization: The next step is to tokenize the words in the data so that they can be used to train the models.
4. Model training: The models are trained using the processed and cleaned data. The models used in this project are LSTM, BERT, BART, and ELECTRA.
5. Frontend development: The final step is to develop the frontend of the app using JavaScript and HTML, which allows users to input words and receive next word predictions.

## Prerequisites
Before running the project, make sure you have installed the following:
- Python and its dependencies, as specified in the `requirements.txt` file. You can install the dependencies by running `pip install -r requirements.txt`.

## Running the project
To run the project, follow these steps:
1. Start the Django server by running `python3 manage.py runserver` from the `nextwordpredictor/` directory.
2. Navigate to `http://127.0.0.1:8000/index` in your browser.

## Using the Web Interface
The web interface allows you to input a word or words and receive suggestions for the next word. Simply type in your word(s) and hit the "Predict" button. The predictions from the four models will be displayed in the results section.

<img src="https://github.com/vaishanth-rmrj/Next-Word-Predictor-LSTM-BERT-BART-ELECTRA/blob/main/extras/next_word_pred_app_gui.png" width=800/>

## Conclusion
This project demonstrates the implementation of a next word prediction app using LSTM, BERT, BART, and ELECTRA models with a Django web interface. The project provides a complete end-to-end solution for next word prediction and can be used as a reference for other similar projects

## Contribution
If you would like to contribute to this project, feel free to create a pull request with your changes. We appreciate any contributions that improve the functionality and performance of the Next-Word-Predictor-LSTM-BERT-BART-ELECTRA app.
