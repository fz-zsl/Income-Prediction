import argparse
import os
import pickle

import pandas as pd
from keras.src.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='train')
	args = parser.parse_args()

	print("Loading dataset ...")
	with open(os.path.join('data', args.dataset, 'train.pkl'), 'rb') as f:
		train_set = pickle.load(f)
	train_df = pd.DataFrame(train_set)
	X = train_df.drop('label', axis=1)
	y = train_df['label']
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

	print("Building Neural Network ...")
	SIZE = 128
	model = Sequential()
	model.add(Dense(SIZE, activation='relu', input_shape=(X_train.shape[1],)))
	model.add(Dropout(0.5))
	while SIZE > 32:
		model.add(Dense(SIZE, activation='relu'))
		model.add(Dropout(0.5))
		SIZE //= 2
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	checkpoint = ModelCheckpoint(os.path.join('checkpoints', f'{args.dataset}.h5'), save_best_only=True)

	print("Training Neural Network ...")
	history = model.fit(
		X_train, y_train, epochs=50, batch_size=64, verbose=1,
		validation_data=(X_val, y_val), callbacks=[checkpoint],
	)
