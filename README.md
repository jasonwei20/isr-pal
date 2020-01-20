# Deep Learning for Bias Detection in the Israeli-Palestinian Conflict

This is the code for training various text classifiers on detecting bias for texts on the Israeli-Palestinian conflict. 
Data for the ip-news dataset is here: https://github.com/jasonwei20/isr-pal/tree/master/ipnews-dataset

Models supported (see `model_types` in `config.py`): 
* Logistic Regression 
* Recurrent Neural Network
* Convolutional Neural Network

Data augmentation methods (see `aug_types` in `config.py`):
* Synonym replacement 
* Sliding window

## Usage

### 1. Word embeddings

Download [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/) and place them in the master folder.

### 2. Load the word embeddings for your vocabulary:
```
python gen_dicts.py
```

### 3. Train logistic regression, convolutional neural network, and recurrent neural networks:
```
python train.py
```

### 2. Evaluate all models for precision, recall, and F1 score:
```
python eval.py
```
