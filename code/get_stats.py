import math
import time
import numpy as np
from random import randint
import datetime

import os
from os import listdir
from os.path import isfile, join, isdir

#get full image paths
def get_image_paths(folder):
	image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
	if join(folder, '.DS_Store') in image_paths:
		image_paths.remove(join(folder, '.DS_Store'))
	image_paths = sorted(image_paths)
	return image_paths

def get_num_words_and_vocab(file):

	num_words = 0
	vocab = set()

	lines = open(file, 'r').readlines()
	for line in lines:
		word = line[:-1].split(' ')
		for word in words:
			num_words += 1
			vocab.add(word)

	print(file, num_words, len(vocab))

	return num_words, vocab

