import argparse
import csv
import os.path
import pickle
import pandas as pd

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='train')
	parser.add_argument('--sep', action='store_true')
	args = parser.parse_args()

	with open(f"{args.dataset}data.csv", 'r') as f:
		reader = csv.DictReader(f)
		data = [_ for _ in reader]

	if args.sep:
		train_size = len(data)
		with open(f"{args.dataset}label.txt", 'r') as f:
			labels = [int(row) for row in f]
		with open(f"testdata.csv", 'r') as f:
			reader = csv.DictReader(f)
			test_data = [_ for _ in reader]
		test_size = len(test_data)
		data += test_data
		labels += [0] * test_size
		for row, label in zip(data, labels):
			row['label'] = int(label)
	else:
		train_size = int(len(data) * 0.9)
		for row in data:
			row['label'] = 1 if row['label'].strip() == '>50K' else 0

	for row in data:
		row['age'] = int(row['age'].strip())
		row['workclass'] = row['workclass'].strip()
		row['fnlwgt'] = int(row['fnlwgt'].strip())
		row['education'] = row['education'].strip()
		row['education_num'] = int(row['education.num'].strip())
		del row['education.num']
		row['marital_status'] = row['marital.status'].strip()
		del row['marital.status']
		row['occupation'] = row['occupation'].strip()
		row['relationship'] = row['relationship'].strip()
		row['race'] = row['race'].strip()
		row['sex'] = row['sex'].strip()
		row['capital_gain'] = int(row['capital.gain'].strip())
		del row['capital.gain']
		row['capital_loss'] = int(row['capital.loss'].strip())
		del row['capital.loss']
		row['hours_per_week'] = int(row['hours.per.week'].strip())
		del row['hours.per.week']
		row['native_country'] = row['native.country'].strip()
		del row['native.country']

	data_df = pd.DataFrame(data)
	data_enc = pd.get_dummies(data_df)

	train_set = data_enc[:train_size]
	test_set = data_enc[train_size:]

	if not os.path.exists(args.dataset):
		os.makedirs(args.dataset)
	with open(os.path.join(args.dataset, 'train.pkl'), 'wb') as f:
		pickle.dump(train_set, f)
	with open(os.path.join(args.dataset, 'test.pkl'), 'wb') as f:
		pickle.dump(test_set, f)
