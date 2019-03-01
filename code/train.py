
import config
import models
from utils import *

word2vec = pickle.load(open(config.pickle_path, 'rb'))
x_dev, y_dev = get_dev_matrices(config.i_dev, config.p_dev, config.sentence_length, config.word2vec_len, word2vec)

#train the model
for aug_type in config.aug_types:

	#inputs
	i_train = '/'.join([config.dataset, 'data', 'train', 'i_' + aug_type + '.txt'])
	p_train = '/'.join([config.dataset, 'data', 'train', 'p_' + aug_type + '.txt'])

	#get training data
	x_train, y_train = get_train_matrices(i_train, p_train, config.sentence_length, config.word2vec_len, word2vec)

	#instantiate the model
	for model_type in config.model_types:

		#outputs
		checkpoint_path = '/'.join([config.dataset, 'checkpoints', model_type + '_' + aug_type + '.h5'])

		#build the model
		model = models.build_model(model_type, 'w2v', config.sentence_length, config.word2vec_len)

		#train the model
		models.train_model(model, x_train, y_train, x_dev, y_dev, checkpoint_path)
