from utils import *
#0=i, 1=p

sentence_length = 45
word2vec_len = 300
huge_word2vec = 'glove.840B.300d.txt'

### general
dataset = 'ipnews'
pickle_path = join(dataset, 'all_vocab.p')


### training
aug_types = ['no_aug', 'sr', 'sw', 'sw_sr']
model_types = ['lr', 'cnn', 'rnn']

i_dev, p_dev = '/'.join([dataset, 'data', 'i_dev.txt']), '/'.join([dataset, 'data', 'p_dev.txt'])

### testing

if dataset == 'sbs':
	i_test, p_test = '/'.join([dataset, 'data', 'i_test.txt']), '/'.join([dataset, 'data', 'p_test.txt'])

elif dataset == 'ipnews':
	i_test, p_test = '/'.join([dataset, 'data', 'i_test']), '/'.join([dataset, 'data', 'p_test'])













