
from utils import *

def build_model(model_type, embedding_type, sentence_length, word2vec_len):

	if embedding_type == "w2v":

		if model_type == "lr":
			return build_w2v_lr(sentence_length, word2vec_len)

		elif model_type == "cnn":
			return build_w2v_cnn(sentence_length, word2vec_len)

		elif model_type == "rnn":
			return build_w2v_rnn(sentence_length, word2vec_len)

		else:
			return None

	return None

def build_w2v_lr(sentence_length, word2vec_len):
	model = None
	model = Sequential()
	model.add(Flatten(input_shape=(sentence_length, word2vec_len)))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def build_w2v_cnn(sentence_length, word2vec_len):
	model = None
	model = Sequential()
	model.add(layers.Conv1D(128, 5, activation='relu', input_shape=(sentence_length, word2vec_len)))
	model.add(layers.GlobalMaxPooling1D())
	model.add(Dense(20, activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def build_w2v_rnn(sentence_length, word2vec_len):
	model = None
	model = Sequential()
	model.add(Bidirectional(LSTM(sentence_length, return_sequences=True), input_shape=(sentence_length, word2vec_len)))
	model.add(Dropout(0.5))
	model.add(Bidirectional(LSTM(sentence_length, return_sequences=False)))
	model.add(Dropout(0.5))
	model.add(Dense(20, activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def train_model(model, x_train, y_train, x_dev, y_dev, out_path):
	callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

	model.fit(	x_train, 
				y_train, 
				batch_size=1024, 
				epochs=100000, 
				callbacks=callbacks,
				validation_data=(x_dev, y_dev),  
				shuffle=True)

	model.save(out_path)








