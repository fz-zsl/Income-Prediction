import argparse
import os
import pickle

import pandas as pd
from keras import Sequential
from keras.src.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='full')
	args = parser.parse_args()

	with open(os.path.join('data', args.dataset, 'test.pkl'), 'rb') as f:
		test_set = pickle.load(f)

	test_df = pd.DataFrame(test_set)
	y_test = test_df['label']
	X_test = test_df.drop('label', axis=1)
	scaler = StandardScaler()
	X_test_scaled = scaler.fit_transform(X_test)

	SIZE = 128
	model = Sequential()
	model.add(Dense(SIZE, activation='relu', input_shape=(X_test.shape[1],)))
	model.add(Dropout(0.5))
	while SIZE > 32:
		model.add(Dense(SIZE, activation='relu'))
		model.add(Dropout(0.5))
		SIZE //= 2
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	ckpt_file = os.path.join('checkpoints', f'{args.dataset}.h5')
	model.load_weights(ckpt_file)
	predictions = (model.predict(X_test_scaled).ravel() >= 0.5).astype(int)
	loss, accuracy = model.evaluate(X_test_scaled, y_test)
	print(f'Neural Network Loss: {loss}, Accuracy: {accuracy}')

	cm = confusion_matrix(y_test, predictions)
	TN = cm[0, 0]
	FP = cm[0, 1]
	FN = cm[1, 0]
	TP = cm[1, 1]
	print(f"TP = {TP}")
	print(f"FP = {FP}")
	print(f"FN = {FN}")
	print(f"TN = {TN}")
