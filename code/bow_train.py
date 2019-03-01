
import config
from utils import *

word_to_idx = load_pickle(join(config.dataset, "bow_word_to_idx.p"))
x_dev, y_dev = get_bow_matrices(config.i_test, config.p_test, word_to_idx)
x_test, y_test = get_bow_matrices(config.i_test, config.p_test, word_to_idx)

#train the model
for aug_type in config.aug_types[:2]:

	#inputs
	i_train = '/'.join([config.dataset, 'data', 'train', 'i_' + aug_type + '.txt'])
	p_train = '/'.join([config.dataset, 'data', 'train', 'p_' + aug_type + '.txt'])

	#get data
	x_train, y_train = get_bow_matrices(i_train, p_train, word_to_idx)

	#tree_clf = tree.DecisionTreeClassifier()
	clf = GaussianNB() #SVC(gamma='auto')
	clf.fit(x_train, y_train) 
	y_pred_train = clf.predict(x_train)
	print(get_metrics(y_train, y_pred_train))
	y_pred_dev = clf.predict(x_dev)
	print(y_pred_dev)
	print(y_dev)
	print(get_metrics(y_dev, y_pred_dev))
