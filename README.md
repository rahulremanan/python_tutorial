# Learn Python 3 for NLP using deep learning.

**A repository of codes and tutorials for running natural language processing (NLP) using deep learning methods with Keras and Tensorflow.**

**For the "Hello World! Python Workshops @ Think Coffee", intermediate complexity lessons organized by the AI @ Columbia University Medical Center NYC MeetUp co-organizer Rahul Remanan.**

Instructions for running 05_Bidirectional_LSTM_classifier.py:

	Install spacy english language model:
	$sudo python -m spacy download en
	
	Create a folder to save the trained model:
	$mkdir model

	Train the sentiment analysis model:
	$python 05_Bidirectional_LSTM_classifier.py model train test
