import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score

start = time.clock()

#We have a dependent discrete variable, which is binary : psoriasis or not psoriasis
#This is a classification problem and will be tackled with the appropriate
#algorithms

train_file = 'GDS4602Give.csv'
#Validation_file = 'GDS4602Validation.csv'
train_file = 'GDS4431Give.csv'
train_file = 'SRP062966Give.csv'
train_file = 'SRP035988Give.csv'
#train_file=input('State your file name: ')

#Load the files as numpy arrays
train_data = np.genfromtxt(train_file, dtype=str, delimiter=',')
#Validation_data = np.genfromtxt(Validation_file, dtype=str, delimiter=',')

#remove titles
train_data = train_data[1:,1:]

np.random.shuffle(train_data)

#Keep the data matrix and the label vector separate
X = train_data[:,:-1].astype(float)
y = train_data[:,-1]

#Convert labels to one hot format
from sklearn.preprocessing import LabelBinarizer
onehot = LabelBinarizer()
y = onehot.fit_transform(y).reshape(y.shape[0],)
#The reshape is to avoid errors, because cross validation in scikit learn
#requires (90,) instead of (90,1) although they are the same

print(time.clock()-start)
#====================================================================
#========================= Data Visualization =======================
#====================================================================

class Visualization:

	def __init__(self, explained_variance, plot_data, eigen_vecs, eigen_vals):
		self.explained_variance = explained_variance
		self.plot_data = plot_data
		self.eigen_vecs = eigen_vecs
		self.eigen_vals = eigen_vals

	@classmethod
	def Decomposition(cls, X):
		U,S,V = np.linalg.svd(X.T, full_matrices=False)
		cls.eigen_vecs=U.T
		cls.eigen_vals=S

	def explained_variance():
		tot = sum(Visualization.eigen_vals)
		var_exp = [(i / tot) for i in sorted(Visualization.eigen_vals, reverse=True)]
		cum_var_exp = np.cumsum(var_exp)
		plt.bar(range(X.shape[0]), var_exp, alpha=0.5, align='center', label='individual explained variance')
		plt.step(range(X.shape[0]), cum_var_exp, where='mid', label='cumulative explained variance')
		plt.ylabel('Explained variance ratio')
		plt.xlabel('Principal components')
		plt.legend(loc='best')
		plt.show()

	def plot_data3D():
		from mpl_toolkits.mplot3d import Axes3D

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		projected=Visualization.eigen_vecs[:3].dot(X.T)
		ax.scatter(projected[0],projected[1],projected[2], c=y)

		ax.set_xlabel('PC1')
		ax.set_ylabel('PC2')
		ax.set_zlabel('PC3')
		plt.show()

	def plot_data():

		fig = plt.figure()
		ax = fig.add_subplot(111)

		projected=Visualization.eigen_vecs[:2].dot(X.T)
		ax.scatter(projected[0],projected[1], c=y)

		ax.set_xlabel('PC1')
		ax.set_ylabel('PC2')
		plt.show()

Visualization.Decomposition(X)
Visualization.explained_variance()
Visualization.plot_data()
Visualization.plot_data3D()

#==================================================================
#======= Define The Repeated Nested Cross Validation ================================
#==================================================================


def RepeatedNCV(gs, X=X, y=y, K=5, rep=10):

	def NestedCV():

		from sklearn.model_selection import StratifiedKFold

		skf = StratifiedKFold(n_splits=K, shuffle=True)
		accuracy_list = []
		parameter_list = []

		#In each iteration 1 fold will be the test set and the rest
		#will be the train set. The number of iterations will equal
		#the number of folds.
		for train_index, test_index in skf.split(X, y):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]

			#Fit to the classifier specified by grid search
			#gridsearch performs the innermost cross validation
			gs.fit(X_train, y_train)
			print('Cross Validation Score:')
			print(gs.best_score_)
			print('With best parameters:')
			print(gs.best_params_)

			y_pred = gs.predict(X_test)
			accu = accuracy_score(y_test, y_pred)
			accuracy_list.append(accu)
			parameter_list.append(gs.best_params_)
		print('The accuracies on test sets were:', accuracy_list)
		#Output the mean and st deviation of the absolute error
		return(np.mean(accuracy_list), np.std(accuracy_list), parameter_list)

	final_accuracy_list=[]
	first = NestedCV()
	parameter_list = first[2]
	final_accuracy_list.append(first[0])

	for i in range(rep-1):
		final_accuracy_list.append(NestedCV()[0])

	return(np.mean(final_accuracy_list), np.std(final_accuracy_list), parameter_list )



#==================================================================
#========================= Model Selection ========================
#==================================================================

file = open("ModelSelectionAndValidation.txt","w")


#----------------Importing Libraries-----------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import f_classif, SelectKBest, VarianceThreshold

#------------- Define a general Pipeline -------------

scaler = MinMaxScaler()
anova = SelectKBest(f_classif, k=X.shape[0])
select = Pipeline([('filter', VarianceThreshold(threshold=0.0001)), ('aov', anova)])

def Pipe(classifier, selector=select, scaler=scaler):
	if selector == None:
		pipe = Pipeline([('scl', scaler), ('clf', classifier)])
	else:
		pipe = Pipeline([('scl', scaler), ('sel', selector), ('clf', classifier)])
	return(pipe)

#-----------------------Naive_Bayes-------------------------

from sklearn.naive_bayes import GaussianNB

pipe_nb = Pipe(GaussianNB(), selector=select)

param_grid={'sel__aov__k':[90,80,50,30]}
gs = GridSearchCV(estimator=pipe_nb, param_grid=param_grid, cv=5)

print('Naive_Bayes yielded the following results:')
FinalScore=RepeatedNCV(gs=gs)
print('Nested Cross Validation Accuracy Score:')
print(FinalScore[0], '+/-',FinalScore[1])

#--------------------------KNN-----------------------------

from sklearn.neighbors import KNeighborsClassifier

pipe_knn = Pipe(KNeighborsClassifier())

param_range = [0.01, 1]
param_grid = {'clf__metric':['euclidean'], 'clf__n_neighbors':range(5,15,3), 'clf__weights':['distance'],  'clf__n_jobs':[-1]}

#For KNN th e lower the K the more complex the model and the more likely we are
#to overfit. The higher the K the more likely we are to underfit

#GridSearch performs stratified KFold validation using a variety of parameters. The data is basically split in K folds and for each iteration K-1 folds are used as training set, while the other 1 is used as test set. The scaling and the feature extraction are fitted to the train set, so there is no information about the test set when it is predicted.
gs = GridSearchCV(estimator=pipe_knn, param_grid=param_grid, cv=5, n_jobs=-1)

print('KNN yielded the following results:')
FinalScore=RepeatedNCV(gs=gs)
print('Nested Cross Validation Accuracy Score:')
print(FinalScore[0], '+/-',FinalScore[1])

#--------------------------SVM-----------------------------

from sklearn.svm import SVC

#No need to select features beforehand, we apply regularization instead
pipe_svc=Pipe(SVC())

param_range = [0.0001, 0.001, 0.01] #Certainly use a low regularization term. We have a high feature space so we need to filter it.
param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']}, {'clf__C': param_range,'clf__kernel': ['sigmoid','poly'], 'clf__gamma':[0.001, 0.01,0.1,1]}]
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid,scoring='accuracy',cv=5, n_jobs=-1)

print('SVC yielded the following results:')
FinalScore=RepeatedNCV(gs=gs)
print('Nested Cross Validation Accuracy Score:')
print(FinalScore[0], '+/-',FinalScore[1])

#------------------ LogisticRegression -------------------

from sklearn.linear_model import LogisticRegression

pipe_lr = Pipe(LogisticRegression(penalty='l1'))

param_range = [0.01, 0.1, 1.0,10]
param_grid = [{'clf__C': param_range}]

gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid,scoring='accuracy',cv=5, n_jobs=-1)

print('Logistic Regression yielded the following results:')
FinalScore=RepeatedNCV(gs = gs)
print('Nested Cross Validation Accuracy Score:')
print(FinalScore[0], '+/-',FinalScore[1])
