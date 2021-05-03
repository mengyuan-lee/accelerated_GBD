# -*- coding:utf-8 -*-
#--------------------------------------------------------
#main
#classification method
#--------------------------------------------------------

import math
import sys
import numpy as np
import matlab
import matlab.engine
import os
import matplotlib.pyplot as plt


import gbd_single_train as gbd
import gbd_test_acc as gbdTestAcc

import gbd_single as gbdSingle
import gbd_multi as gbdMulti
import json


from sklearn import preprocessing, discriminant_analysis, linear_model, svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_curve, auc 



eng = matlab.engine.start_matlab()

def load_data(path):
	f=open(path,'r',encoding='utf-8')
	data_input = json.load(f)

	features = data_input['features']
	labels = data_input['labels']
	if len(labels) != len(features):
		print("Input data error")
		sys.exit()
	
	x = []
	y = []

	for label in labels:
		y.append(label['VALUABLE'])	

	for feature in features:
		temp = {}

		temp[0] = feature['num_cut']
		temp[1] = feature['i_gbd']
		temp[2] = feature['cut_violation']
		temp[3] = feature['cut_type']
		temp[4] = feature['cut_order']

		x.append(temp)

	return y,x

def load_all_data(K0,L0,train_flag):
	if train_flag == 'train':
		path = 'outputs/problem_'+str(K0)+'_'+str(L0)
	else:
		path = 'test_outputs/problem_'+str(K0)+'_'+str(L0)
	print(os.listdir(path))
	files = os.listdir(path)
	y = []
	x = []
	for file in files:
		temp_y, temp_x = load_data(path+'/'+file)
		y += temp_y
		x += temp_x
	return y,x

def normalize_data(y,x):
	x_feature = np.zeros((len(y),5))
	for i in range(len(y)):
		for j in range(5):
			x_feature[i,j]=x[i][j]
	stand_scaler = preprocessing.StandardScaler()
	x_scaled = stand_scaler.fit_transform(x_feature)
	x_scaled_dict = []
	for i in range(len(y)):
		temp = {}

		temp[0] = x_scaled[i,0]
		temp[1] = x_scaled[i,1]
		temp[2] = x_scaled[i,2]
		temp[3] = x_scaled[i,3]
		temp[4] = x_scaled[i,4]

		x_scaled_dict.append(temp)
	return y,x_scaled_dict, stand_scaler

def under_sample(y,x,del_num):
	i=0
	while i<del_num:
		del_index = np.random.randint(low=0, high=len(y))
		if y[del_index]==0:
			del y[del_index]
			del x[del_index]
			i = i + 1
	return y,x

def dic_to_array(y,x):
	x_feature = np.zeros((len(y),5))
	for i in range(len(y)):
		for j in range(5):
			x_feature[i,j]=x[i][j]
	return x_feature

def num_count(y):
	num1 = 0
	num0 = 0

	for i in range(len(y)):
		if y[i]==1:
			num1 = num1 + 1
		else:
			num0 = num0 + 1

	print(num1, num0)

	return num1, num0


def normalize_data_scaler(y,x,scaler):
	x_feature = np.zeros((len(y),5))
	for i in range(len(y)):
		for j in range(5):
			x_feature[i,j]=x[i][j]
	x_scaled = scaler.transform(x_feature)
	x_scaled_dict = []
	for i in range(len(y)):
		temp = {}

		temp[0] = x_scaled[i,0]
		temp[1] = x_scaled[i,1]
		temp[2] = x_scaled[i,2]
		temp[3] = x_scaled[i,3]
		temp[4] = x_scaled[i,4]

		x_scaled_dict.append(temp)
	return y,x_scaled_dict

def performance_metric(y,x,model):
	model = model.fit(x,y)
	predictions = model.predict(x)
	print('Accuracy:', accuracy_score(y, predictions))  
	print('Recall:', recall_score(y, predictions)) 
	cm = confusion_matrix(y, predictions)
	print('Recall negative:', cm[0,0]/(cm[0,0]+cm[0,1])) 
	#print('Confusion matrix:\n', cm)
	y_score = model.decision_function(x)
	fpr,tpr,threshold = roc_curve(y, y_score) #calculate false_positive and true_positive rate
	roc_auc = auc(fpr,tpr) #calcuate auc

	return fpr, tpr, roc_auc, model

def gbd_run(K0, L0, data_num, numsol, model, train_flag, threshold, scaler):

	mattr='data_'+str(K0)+'_'+str(L0)+'_'+train_flag+'.mat'  
	K0,L0,R_min_C0,P_max_D0,P_max_C0,H_CD0,H_D0,H_CB0,H_DB0= gbd.data_extract(mattr)

	#result file
	res_fold = train_flag+'/problem_'+str(K0)+'_'+str(L0)+'/result'
	if not os.path.exists(res_fold):
		os.makedirs(res_fold)
	log_fold = train_flag+'/problem_'+str(K0)+'_'+str(L0)+'/log'
	if not os.path.exists(log_fold):
		os.makedirs(log_fold)

	res_path = train_flag+'/problem_'+str(K0)+'_'+str(L0)+'/result/result_'+str(numsol)+'.txt'

	if train_flag == 'test':
		log_json_fold = train_flag+'/problem_'+str(K0)+'_'+str(L0)+'/log_json'
		if not os.path.exists(log_json_fold):
			os.makedirs(log_json_fold)
		acc_path = train_flag+'/problem_'+str(K0)+'_'+str(L0)+'/result/acc_'+str(numsol)+'.txt'
		acc_total=0
		num_total=0
		acc1_total=0
		num1_total=0
		acc0_total=0
		num0_total=0

	for i in range(data_num):
		iter_total = 0
		res_file = open(res_path,mode='a+')
		res_file.writelines(['Problem',str(i+1),'\n'])
		res_file.close()
		log_path = train_flag+'/problem_'+str(K0)+'_'+str(L0)+'/log/log_'+str(numsol)+'_problem'+str(i+1)+'.txt'
		if train_flag == 'test':
			log_json_path = train_flag+'/problem_'+str(K0)+'_'+str(L0)+'/log_json/log_'+str(numsol)+'_problem'+str(i+1)+'.json'
		rho_initial = np.zeros((K0,L0))
		repeat_number= 0
		while repeat_number<=1:
			print("solving problem #%d..."%(i+1))
			h_CD0 = H_CD0[:,i].reshape((K0,L0))
			h_D0 = H_D0[:,i]
			h_CB0 = H_CB0[:, i]
			h_DB0 = H_DB0[:, i]


			if train_flag=='train':
				p_D_opt, rho_opt, rho, convergence_flag, iter_num= gbd.gbd_single_train(K0,L0,R_min_C0,P_max_D0,P_max_C0,h_CD0,h_D0,\
					h_CB0,h_DB0,rho_initial,numsol,threshold,res_path,log_path,i)
			elif train_flag == 'test':
				p_D_opt, rho_opt, rho, convergence_flag, iter_num, acc_array = gbdTestAcc.gbd_test_acc(K0,L0,R_min_C0,P_max_D0,P_max_C0,h_CD0,h_D0,\
					h_CB0,h_DB0,rho_initial,numsol,model,res_path,log_path,log_json_path,0.99,i,scaler)
				acc_total=acc_total+acc_array[0]
				num_total=num_total+acc_array[1]
				acc1_total=acc1_total+acc_array[2]
				num1_total=num1_total+acc_array[3]
				acc0_total=acc0_total+acc_array[4]
				num0_total=num0_total+acc_array[5]

			iter_total = iter_total + iter_num
			
			if convergence_flag == 1:
				print("problem #%d solved!"%(i+1))
				res_file = open(res_path,mode='a+')
				res_file.writelines(['total iteration:\t',str(iter_total),'\n'])
				res_file.writelines(['\n'])
				res_file.close()
				break
			else:
				if convergence_flag == 0:
					print("problem #%d cannot converge!"%(i+1))
					res_file = open(res_path,mode='a+')
					res_file.writelines(['total iteration:\t',str(iter_total),'\n'])
					res_file.writelines(['\n'])
					res_file.close()
					break
				else:
					print("new iteration needed!")
					if train_flag == 'train':
						rho_initial = rho.copy()
					else:
						rho_initial = rho[0,:,:].copy()
					res_file = open(res_path,mode='a+')
					res_file.writelines(['total iteration now:\t',str(iter_total),'\n'])
					res_file.writelines(['\n'])
					res_file.close()
					repeat_number = repeat_number + 1
		if repeat_number>1:
			print("problem #%d cannot converge!"%(i+1))
			res_file = open(res_path,mode='a+')
			res_file.writelines(['Problem cannot converge','\n'])
			res_file.writelines(['total iteration:\t',str(iter_total),'\n'])
			res_file.writelines(['\n'])
			res_file.close()

		if train_flag == 'test':
			acc_file = open(acc_path,mode='a+')
			acc_file.writelines(['total accuracy after %d problem:'%(i+1),'\n'])
			acc_file.writelines(['acc:',str(acc_total/num_total),'\tacc1:\t',str(acc1_total/num1_total),'\tacc0:\t', str(acc0_total/num0_total),'\n'])
			acc_file.writelines(['\n'])
			acc_file.close()


def main(argv=None):
	#######################################
	#hyper parameters
	#for training
	K0 = 5
	L0 = 3
	data_num = 50
	numsol = 8
	threshold = 1
	#for testing
	K_test = 6
	L_test = 4
	data_num_test = 50
	numsol_test = 8

	scaler_filename = 'model/scaler_'+str(K0)+'_'+str(L0)+'.save'
	model_path_svm = 'model/model_svm_'+str(K0)+'_'+str(L0)+'.m'
	model_path_lda = 'model/model_lda_'+str(K0)+'_'+str(L0)+'.m'
	model_path_qda = 'model/model_qda_'+str(K0)+'_'+str(L0)+'.m'
	model_path_lr = 'model/model_lr_'+str(K0)+'_'+str(L0)+'.m'
	######################################
	#1.collect training dataset
	gbd_run(K0, L0, data_num, numsol,'','train', threshold, '')
	######################################
	#2.read .json file
	y, x = load_all_data(K0,L0,'train')
	#data processing
	num1, num0 = num_count(y)
	y, x = under_sample(y,x, num0-num1)
	num1, num0 = num_count(y)

	y,x_scaled,scaler = normalize_data(y,x)
	x_array = dic_to_array(y,x)
	x_scaled_array = dic_to_array(y,x_scaled)

	np.savetxt('train/problem_'+str(K0)+'_'+str(L0)+'/train_feature.txt', x_array)
	np.savetxt('train/problem_'+str(K0)+'_'+str(L0)+'/train_feature_scale.txt', x_scaled_array)
	np.savetxt('train/problem_'+str(K0)+'_'+str(L0)+'/train_label.txt', y)
	if not os.path.exists('model'):
		os.makedirs('model')
	joblib.dump(scaler, scaler_filename)
	

	#3.training
	#############################################
	# Using for re-train
	# scaler = joblib.load(scaler_filename) 
	# x_array = np.loadtxt('train/problem_'+str(K0)+'_'+str(L0)+'/train_feature.txt')
	# x_scaled_array = np.loadtxt('train/problem_'+str(K0)+'_'+str(L0)+'/train_feature_scale.txt')
	# y = np.loadtxt('train/problem_'+str(K0)+'_'+str(L0)+'/train_label.txt')
	#############################################
	#SVM
	print('SVM:')
	clf = svm.SVC(class_weight={0:1,1:2})
	fpr1, tpr1, roc_auc1, clf = performance_metric(y,x_scaled_array,clf)
	#############################################

	#############################################
	#LDA
	print('\nLDA:')
	lda = discriminant_analysis.LinearDiscriminantAnalysis(priors=[1/3,2/3])
	fpr2, tpr2, roc_auc2, lda = performance_metric(y,x_scaled_array,lda)
	#############################################

	#############################################
	#QDA
	print('\nQDA:')
	qda = discriminant_analysis.QuadraticDiscriminantAnalysis(priors=[1/3,2/3])
	fpr3, tpr3, roc_auc3, qda = performance_metric(y,x_scaled_array,qda)
	#############################################

	#############################################
	#logistic
	print('\nLogitisic:')
	lr = linear_model.LogisticRegressionCV(class_weight={0:1,1:2})
	fpr4, tpr4, roc_auc4, lr = performance_metric(y,x_scaled_array,lr)
	#############################################

	#############################################
	#ROC
	plt.figure()
	plt.grid()
	l1, = plt.plot(fpr1, tpr1, color='sienna', linewidth=2.0, label='SVM(area = %0.2f)' % roc_auc1)
	l2, = plt.plot(fpr2, tpr2, color='red', linewidth=2.0, label='LDA(area = %0.2f)' % roc_auc2)
	l3, = plt.plot(fpr3, tpr3, color='purple', linewidth=2.0, label='QDA(area = %0.2f)' % roc_auc3)
	l4, = plt.plot(fpr4, tpr4, color='green', linewidth=2.0, label='Logistic Regression(area = %0.2f)' % roc_auc4)
	plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.show()
	#############################################

	#############################################
	#4.save model 
	joblib.dump(clf, model_path_svm)
	joblib.dump(lda, model_path_lda)
	joblib.dump(qda, model_path_qda)
	joblib.dump(lr, model_path_lr)
	#############################################

	#############################################
	
	#4' read model and scaler
	scaler = joblib.load(scaler_filename) 
	clf = joblib.load(model_path_svm)
	lda = joblib.load(model_path_lda)   
	qda = joblib.load(model_path_qda)
	lr = joblib.load(model_path_lr)
	#############################################
	
	#############################################
	#5. test
	gbd_run(K_test, L_test, data_num_test, numsol_test, clf, 'test', threshold, scaler)
	gbdMulti.gbd_multi_run(K_test,L_test,numsol_test,data_num_test)
	gbdSingle.gbd_single_run(K_test,L_test,data_num_test)
	



if __name__ == "__main__":
	main()


	