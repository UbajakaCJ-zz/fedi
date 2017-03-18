import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def main():
	# np.random.seed(0)
	print('Loading data...')
	training = pd.read_excel('tournament.xlsx')
	tournament = pd.read_excel('training.xlsx')

	X = training.drop(['Name','Remarks'], axis=1)
	# X.reshape(-1,1)
	Y = training['Remarks']
	# Y.reshape(-1,1)


	t_name = tournament['Name']

	x_predict = tournament.drop(['Name', 'Remarks'], axis=1)
	y_predict = tournament['Remarks']

	# features_train, labels_train, features_test, labels_test = cross_validation.train_test_split(X,Y, test_size=0.3, random_state=0)

	model = DecisionTreeClassifier(min_samples_split=5)

	print('Training...')
	model.fit(X,Y)

	print('Predicting...')
	# pred = model.predict(features_test)

	# accuracy = accuracy_score(pred, labels_test)
	# print('Accuracy:', accuracy)

	y_prediction = model.predict(x_predict)

	accuracy = accuracy_score(y_prediction, y_predict)
	print('Accuracy:', accuracy)

	results_df = pd.DataFrame(data={'Remarks':y_prediction})

	joined = pd.DataFrame(t_name).join([tournament['Average'],results_df])
	joined.loc[joined["Remarks"] == 0, "Remarks"] = 'FAIL'
	joined.loc[joined["Remarks"] == 1, "Remarks"] = 'PASS'


	print("Writing predictions to predictions.csv")
	# Save the predictions out to a CSV file
	joined.to_csv("predictions.csv", index=False)









if __name__ == '__main__':
	main()