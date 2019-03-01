
import config
import models
from utils import *

#load the vocab pickle
word2vec = pickle.load(open(config.pickle_path, 'rb'))

#load the testing data
x_test, y_test = None, None
if config.dataset == 'sbs':
	x_test, y_test = get_test_matrices(config.i_test, config.p_test, config.sentence_length, config.word2vec_len, word2vec)


print("pre\trec\tf1")

#test the models
for aug_type in config.aug_types:

	for model_type in config.model_types:

		#load the model
		checkpoint_path = '/'.join([config.dataset, 'checkpoints', model_type + '_' + aug_type + '.h5'])
		model = load_model(checkpoint_path)

		if config.dataset == 'sbs':
			#get the model's predictions
			y_pred = np.squeeze(model.predict(x_test)).tolist()
			y_pred_binary = conf_to_pred(y_pred)

			#print the metrics
			print(checkpoint_path, get_metrics(y_test, y_pred_binary))

		elif config.dataset == 'ipnews':

			y_test, y_pred = test_ipnews(model, config.i_test, config.p_test, config.word2vec_len, word2vec, config.sentence_length)
			y_pred_binary = conf_to_pred(y_pred)
			print(checkpoint_path, get_metrics(y_test, y_pred_binary))