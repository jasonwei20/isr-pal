import time, os, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #get rid of warnings

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn import tree
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB


from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional, Flatten
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping
import keras.layers as layers

import math
import time
import numpy as np
from random import randint
import datetime, re, operator

import os
from os import listdir
from os.path import isfile, join, isdir
import pickle

from matplotlib import pyplot
from matplotlib import rc

###################################################
######### loading folders and txt files ###########
###################################################

#loading a pickle file
def load_pickle(file):
	return pickle.load(open(file, 'rb'))

#create an output folder if it does not already exist
def confirm_output_folder(output_folder):
	if not os.path.exists(output_folder):
	    os.makedirs(output_folder)

#get full txt paths
def get_txt_paths(folder):
    txt_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and '.txt' in f]
    if join(folder, '.DS_Store') in txt_paths:
        txt_paths.remove(join(folder, '.DS_Store'))
    txt_paths = sorted(txt_paths)
    return txt_paths

#get subfolders
def get_subfolder_paths(folder):
    subfolder_paths = [join(folder, f) for f in listdir(folder) if (isdir(join(folder, f)) and '.DS_Store' not in f)]
    if join(folder, '.DS_Store') in subfolder_paths:
        subfolder_paths.remove(join(folder, '.DS_Store'))
    subfolder_paths = sorted(subfolder_paths)
    return subfolder_paths

#get all image paths
def get_all_txt_paths(master_folder):

    all_paths = []
    subfolders = get_subfolder_paths(master_folder)
    if len(subfolders) >= 1:
        for subfolder in subfolders:
            all_paths += get_txt_paths(subfolder)
    all_paths += get_txt_paths(master_folder)
    return all_paths

###################################################
############## data pre-processing ################
###################################################

#get the pickle file for the vocab so you don't have to load the entire dictionary
def gen_vocab_dicts(folder, output_pickle_path, huge_word2vec):

    vocab = set()
    text_embeddings = open(huge_word2vec, 'r').readlines()
    word2vec = {}

    #get all the vocab
    all_txt_paths = get_all_txt_paths(folder)
    print(all_txt_paths)

    #loop through each text file
    for txt_path in all_txt_paths:

        # get all the words
        all_lines = open(txt_path, "r").readlines()
        for line in all_lines:
            words = line[:-1].split(' ')
            for word in words:
                vocab.add(word)
    
    print(len(vocab), "unique words found, including typos.")

    # load the word embeddings, and only add the word to the dictionary if we need it
    for line in text_embeddings:
        items = line.split(' ')
        word = items[0]
        if word in vocab:
            vec = items[1:]
            word2vec[word] = np.asarray(vec, dtype = 'float32')
    print(len(word2vec), "matches between vocab and word2vec.")
        
    pickle.dump(word2vec, open(output_pickle_path, 'wb'))
    print("dictionaries outputted to", output_pickle_path)

###################################################
################ data processing ##################
###################################################

#balancing a dataset
def balance(isr_lines, pal_lines):
    if len(isr_lines) > len(pal_lines):
        pal_lines = pal_lines + pal_lines
        pal_lines = pal_lines[:len(isr_lines)] 
    elif len(isr_lines) < len(pal_lines):
        isr_lines = isr_lines + isr_lines
        isr_lines = isr_lines[:len(pal_lines)]
    return isr_lines, pal_lines

#putting isr_lines and pal_lines into a numpy array
def get_matrix_from_lines(num_words, word2vec_len, isr_lines, pal_lines, word2vec):
    
    n_isr = len(isr_lines)
    n_pal = len(pal_lines)
    x_matrix = np.zeros((n_isr+n_pal, num_words, word2vec_len))
    
    #add isr lines first
    for i, line in enumerate(isr_lines):
        words = line[:-1].split(' ')
        words = words[:x_matrix.shape[1]]
        for j, word in enumerate(words):
            if word in word2vec:
                x_matrix[i, j, :] = word2vec[word]
    
    #then add pal lines
    for i, line in enumerate(pal_lines):
        words = line[:-1].split(' ')
        words = words[:x_matrix.shape[1]]
        for j, word in enumerate(words):
            if word in word2vec:
                x_matrix[i+n_isr, j, :] = word2vec[word]

    y_matrix = np.zeros(n_isr+n_pal)
    y_matrix[n_isr:] = 1
    
    return x_matrix, y_matrix

def get_train_matrices(i_train_path, p_train_path, sentence_length, word2vec_len, word2vec):

    isr_train = open(i_train_path, 'r').readlines()
    pal_train = open(p_train_path, 'r').readlines() 
    isr_train, pal_train = balance(isr_train, pal_train) 
    print("training:", len(isr_train), 'isr lines loaded and', len(pal_train), 'pal lines loaded')
    x_train, y_train = get_matrix_from_lines(sentence_length, word2vec_len, isr_train, pal_train, word2vec)
    print("training matrix shapes:", x_train.shape, y_train.shape)
    return x_train, y_train

def get_dev_matrices(i_dev_path, p_dev_path, sentence_length, word2vec_len, word2vec):
    isr_dev = open(i_dev_path, 'r').readlines()
    pal_dev = open(p_dev_path, 'r').readlines() 
    print("dev:", len(isr_dev), 'isr lines loaded and', len(pal_dev), 'pal lines loaded')
    x_dev, y_dev = get_matrix_from_lines(sentence_length, word2vec_len, isr_dev, pal_dev, word2vec)
    print("dev matrix shapes:", x_dev.shape, y_dev.shape)
    return x_dev, y_dev

def get_test_matrices(i_test_path, p_test_path, sentence_length, word2vec_len, word2vec):
    isr_test = open(i_test_path, 'r').readlines()
    pal_test = open(p_test_path, 'r').readlines() 
    print("test:", len(isr_test), 'isr lines loaded and', len(pal_test), 'pal lines loaded')
    x_test, y_test = get_matrix_from_lines(sentence_length, word2vec_len, isr_test, pal_test, word2vec)
    print("test matrix shapes:", x_test.shape, y_test.shape)
    return x_test, y_test

def get_bow_from_lines(i_lines, p_lines, word_to_idx):

    n_isr = len(i_lines)
    n_pal = len(p_lines)
    x_matrix = np.zeros((n_isr+n_pal, len(word_to_idx.keys())))

    for i, i_line in enumerate(i_lines):
        add_line_to_x_matrix(i_line, x_matrix, i, word_to_idx)

    for i, p_line in enumerate(p_lines):
        add_line_to_x_matrix(i_line, x_matrix, n_isr + i, word_to_idx)

    y_matrix = np.zeros(n_isr+n_pal)
    y_matrix[n_isr:] = 1

    return x_matrix, y_matrix

def add_line_to_x_matrix(line, x_matrix, line_num, word_to_idx):

    words = line[:-1].split(" ")

    #add one-grams
    for word in words:
        if word in word_to_idx:
            x_matrix[line_num, word_to_idx[word]] += 1.0

    #add two-grams
    for i in range(len(words)-1):
        bigram = words[i] + ' ' + words[i+1]
        if bigram in word_to_idx:
            x_matrix[line_num, word_to_idx[bigram]] += 1.0

def get_bow_matrices(i_train_path, p_train_path, word_to_idx):

    isr_train = open(i_train_path, 'r').readlines()
    pal_train = open(p_train_path, 'r').readlines() 
    isr_train, pal_train = balance(isr_train, pal_train)
    print(len(isr_train), 'isr lines loaded and', len(pal_train), 'pal lines loaded')

    x_matrix, y_matrix = get_bow_from_lines(isr_train, pal_train, word_to_idx)
    return x_matrix, y_matrix




###################################################
################### evaluation ####################
###################################################

#confidences to binary
def conf_to_pred(y):

    if type(y) == list:
        y_class = []
        for pred in y:
            if pred < 0.5:
                y_class.append(0)
            else:
                y_class.append(1)
        return y_class

    else:
        y_class = np.zeros(y.shape)
        for i in range(y.shape[0]):
            if y[i] < 0.5:
                y_class[i] = 0
            else:
                y_class[i] = 1
        return y_class
    
#accuracy of the model
def get_accuracy(model, x, y):
    y_predict = model.predict(x)
    y_class = conf_to_pred(y_predict)
    return accuracy_score(y, y_class)

#cleaning up the file
def get_only_chars(line):

    clean_line = ""

    line = line.lower()
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line)
    return clean_line

def prettify_num(number):
    return str(round(100*number, 1)) 

#cleaning up metrics
def prettify(metric_list):
    return " & ".join([prettify_num(x) for x in metric_list])

def get_metrics(y_test, y_pred_binary):
    metrics = [precision_score(y_test, y_pred_binary), recall_score(y_test, y_pred_binary), f1_score(y_test, y_pred_binary)]
    return prettify(metrics)

#testing a single article
def test_article(model, article_path, word2vec_len, word2vec, sentence_length, stride):
    line = open(article_path, 'r',  encoding='latin-1').read()
    line = get_only_chars(line) #clean up the chars
    words = line.split(' ')

    #build up the numpy array
    num_sentences = int( (len(words)-sentence_length) / stride)
    x = np.zeros((num_sentences, sentence_length, word2vec_len))
    sentences = []
    for i in range(num_sentences):
        window_words = words[i*stride:i*stride+sentence_length]
        sentences.append(" ".join(window_words))
        for j, window_word in enumerate(window_words):
            if window_word.lower() in word2vec:
                x[i, j, :] = word2vec[window_word.lower()]
    
    #make the predictions
    y_predict = model.predict(x)

    #return the prediction and the sentence
    return y_predict, sentences

def test_ipnews(model, i_folder, p_folder, word2vec_len, word2vec, sentence_length):

    articles = get_txt_paths(i_folder) + get_txt_paths(p_folder)
    y_test = [0 for _ in range(len(get_txt_paths(i_folder)))] + [1 for _ in range(len(get_txt_paths(p_folder)))]

    article_preds = []
    for article in articles:
        scores, sentences = test_article(model, article, word2vec_len, word2vec, sentence_length, stride=5)
        article_pred = np.mean(scores)
        article_preds.append(article_pred)

    return y_test, article_preds

###################################################
################### freq dist #####################
###################################################

def add_to_freq_dict(f_dict, word):

    if word in f_dict:
        f_dict[word] += 1
    else:
        f_dict[word] = 1

def get_top_words(f_dict, n):

    sorted_by_value = list(reversed(sorted(f_dict.items(), key=lambda kv: kv[1])))
    top_tuples = sorted_by_value[:n]
    top_words = {top_tuples[i][0]:i for i in range(len(top_tuples))}
    return top_words

def add_text_file_to_f_dict(f_dict, text_file):

    lines = open(text_file, 'r').readlines()
    for line in lines:
        words = line[:-1].split(' ')
        for word in words:
            add_to_freq_dict(f_dict, word) 
        for i in range(len(words)-1):
            word_1 = words[i]
            word_2 = words[i+1]
            bigram = ' '.join([word_1, word_2])
            add_to_freq_dict(f_dict, bigram) 

def add_folder_to_f_dict(f_dict, folder):

    all_txt_paths = get_all_txt_paths(folder)
    for txt_path in all_txt_paths:
        add_text_file_to_f_dict(f_dict, txt_path)


# #plotting the prediction distribution
# def plot_preds_dist_news(isr_preds, pal_preds, output_path):

#     bins = np.linspace(0, 1, 20)
#     pyplot.clf()
#     fig, ax = pyplot.subplots()
#     pyplot.rc('font',family='Arial')
#     pyplot.hist([isr_preds, pal_preds], bins, label=['Israeli Origin', 'Palestinian Origin'], color=[(0, 0, 0.5), (1, 0, 0)] )
#     pyplot.xlabel("Predicted Value", fontname="Arial", size=13)
#     pyplot.ylabel("Frequency", fontname="Arial", size=13)
#     pyplot.title("Distribution of Predicted Labels for Newspaper Articles", fontname="Arial", size=16)
#     pyplot.legend(loc='upper right')
#     pyplot.savefig(output_path, dpi=400)

# #plotting the prediction distribution
# def plot_preds_dist_side(isr_preds, pal_preds, output_path):

#     bins = np.linspace(0, 1, 20)
#     pyplot.clf()
#     fig, ax = pyplot.subplots()
#     pyplot.rc('font',family='Arial')
#     pyplot.hist([isr_preds, pal_preds], bins, label=['Israeli Origin', 'Palestinian Origin'], color=[(0, 0, 0.5), (1, 0, 0)] )
#     pyplot.xlabel("Predicted Value", fontname="Arial", size=13)
#     pyplot.ylabel("Frequency", fontname="Arial", size=13)
#     pyplot.title("Distribution of Predicted Labels for Side by Side Excerpts", fontname="Arial", size=16)
#     pyplot.legend(loc='upper right')
#     pyplot.savefig(output_path, dpi=400)

