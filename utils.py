import matplotlib.pyplot as plt 
import numpy as np
import itertools
from sklearn.metrics import roc_auc_score

def plot_roc(fpr, tpr, title='ROC CURVE'):
	'''

	This Function plots the ROC curve for a given False Positive Rate and True Positive Rate

	args:
	- fpr float: False Positive Rate of Model
	- tpr float: True Positive Rate of Model
	- title string: title of ROC
	- ax object: axis for plotting

	'''
	plt.figure(figsize=(12,8))
	plt.title(title, fontsize=16)
	plt.plot(fpr, tpr, 'b-', linewidth=2)
	plt.plot([0, 1], [0, 1], 'r--')
	plt.xlabel('False Positive Rate', fontsize=16)
	plt.ylabel('True Positive Rate', fontsize=16)
	plt.axis([-0.01,1,0,1])


def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	'''

	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.

	Credits for this function: https://www.kaggle.com/joparga3/in-depth-skewed-data-classif-93-recall-acc-now

	'''
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=0)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		#print("Normalized confusion matrix")
	else:
		1 #print('Confusion matrix, without normalization')

	#print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


def plot_roc_multiple(fpr_1, tpr_1, fpr_2, tpr_2,
					   fpr_3, tpr_3, y_test, pred1, pred2, pred3):

	'''

	This Function plots the ROC curve for a given False Positive Rate and True Positive Rate for
	3 different models, We will use this when validating our 3 models in Fraud Detection Analysis.ipynb

	args:
	- fpr_i float: False Positive Rate for ith model
	- tpr_i float: True Positive Rate for ith model
	- y_test df: Labels of testing
	- predi df: Prediction of ith model

	'''
	plt.figure(figsize=(16,8))
	plt.title('ROC Curve \n 3 Classifiers', fontsize=18)
	plt.plot(fpr_1, tpr_1, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_test, pred1)))
	plt.plot(fpr_2, tpr_2, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_test, pred2)))
	plt.plot(fpr_3, tpr_3, label=' 3 Layer Neural Network: {:.4f}'.format(roc_auc_score(y_test, pred3)))
	plt.plot([0, 1], [0, 1], 'k--')
	plt.axis([-0.01, 1, 0, 1])
	plt.xlabel('False Positive Rate', fontsize=16)
	plt.ylabel('True Positive Rate', fontsize=16)
	plt.annotate('Minimum ROC Score of 50% \n (A model below this indicates a worse model than guessing)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
				arrowprops=dict(facecolor='#6E726D', shrink=0.05))
	plt.legend()