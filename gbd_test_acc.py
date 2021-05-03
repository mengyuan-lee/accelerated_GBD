# -*- coding:utf-8 -*-
#machine-learning based multi-cut generalized benders decompostion
#with labels
import sys
import numpy as np
import matlab
import matlab.engine
import master_test_acc as masterTest
import master_multi_train as masterTrain
import scipy.io as sio 
import os
import json


eng = matlab.engine.start_matlab()

def gbd_test_acc(K0,L0,R_min_C0,P_max_D0,P_max_C0,h_CD0,h_D0,h_CB0,h_DB0,rho_initial,numsol,model,res_path,log_path,log_json_path,threshold,i,scaler):
	global K,L,R_min_C,P_max_D,P_max_C,h_CD,h_D,h_CB,h_DB
	K = float(K0)
	L = float(L0)
	R_min_C = float(R_min_C0)
	P_max_D = float(P_max_D0)
	P_max_C = float(P_max_C0)
	h_CD = matlab.double(h_CD0.tolist())
	h_CD.reshape((K0,L0))
	h_D = matlab.double(h_D0.tolist())
	h_D.reshape((L0,1))
	h_CB = matlab.double(h_CB0.tolist())
	h_CB.reshape((K0,1))
	h_DB = matlab.double(h_DB0.tolist())
	h_DB.reshape((L0,1))

	global a, b, p_max
	a, b, p_max = para()

	log_json = []
	

	#set iteration index
	i_gbd = 1
	#set maximum iteration num 
	max_iter_num = 1E5

	#set global upper bound and lower bound
	upper_bound = sys.maxsize
	lower_bound = -sys.maxsize
	#master calss initialization
	master_class_train = masterTrain.Master_class(K0, L0, h_CD0, h_CB0, h_D0, h_DB0, P_max_D0, R_min_C0, P_max_C0, p_max, a, b, i)
	master_class_test = masterTest.Master_class(K0, L0, h_CD0, h_CB0, h_D0, h_DB0, P_max_D0, R_min_C0, P_max_C0, p_max, a, b, i)

	
	alpha = 0

	#avoid local minimum
	reapeat_count = 0
	#optimal rho_result
	rho_opt = np.zeros((K0,L0))
	p_D_opt = np.zeros((K0,L0))

	acc_total = 0
	acc1_total = 0
	acc0_total = 0
	num_total = 0
	num1_total = 0
	num0_total = 0

	time_total = 0
	cut_total = 0

	##########################################################
	#first iteration
	#solve the primal problem 

	feasible_flag, p_D, eta, lameda, miu, nu, theta = primal_solve(rho_initial)
	if feasible_flag==1:
		if upper_bound>-eta:
			rho_opt = rho_initial.copy()
			p_D_opt = np.array(p_D).copy()
			upper_bound = -eta
	else:
		#solve feasibilty check problem 
		p_D, eta, alpha, lameda, miu, nu, theta = infeasible_solve(rho_initial)
		if alpha<1e-20:
			print("find a feasible point")
			feasible_flag=1
			if upper_bound>-eta:
				rho_opt = rho_initial.copy()
				p_D_opt = np.array(p_D).copy()
				upper_bound = -eta
		else:
			print("find an infeasible point")

	#solve the relaxed master problem 
	feasible_flag_multi = np.array(feasible_flag).reshape(1,1)
	p_D_multi = np.array(p_D).reshape(1,K0,L0)
	eta_multi = np.array(eta).reshape(1,1)
	alpha_multi = np.array(alpha).reshape(1,1)
	lameda_multi = np.array(lameda).reshape(1,L0,1)
	miu_multi = np.array(miu).reshape(1,K0,L0)
	nu_multi = np.array(nu).reshape(1,K0,L0)
	theta_multi = np.array(theta).reshape(1,L0,1)

	print("Master problem solving:")
	cut_label = master_class_train.master_sol(feasible_flag_multi, p_D_multi, eta_multi, alpha_multi, lameda_multi,\
		miu_multi, nu_multi, theta_multi, numsol, rho_initial, threshold, lower_bound, rho_initial)

	rho_new, psi, numsol_real, num_addcut, num_acc, num_acc1, num_acc0, this_num, num_1, num_0, time_new = master_class_test.master_sol(feasible_flag_multi, p_D_multi, eta_multi, \
		alpha_multi, lameda_multi, miu_multi, nu_multi, theta_multi, numsol, rho_initial, lower_bound, rho_initial,model,cut_label,scaler)

	acc = num_acc/this_num
	if num_1==0:
		acc1 = 'None'
	else:
		acc1 = num_acc1/num_1
	if num_0==0:
		acc0 = 'None'
	else:
		acc0 = num_acc0/num_0

	acc_total = acc_total + num_acc
	acc1_total = acc1_total + num_acc1
	acc0_total = acc0_total + num_acc0
	num_total = num_total + this_num
	num1_total = num1_total + num_1
	num0_total = num0_total + num_0

	time_total = time_total + time_new
	cut_total = cut_total + num_addcut


	#rho_new.shape (numsol,K,L)
	#psi.shape (1,1) only return incumbent solution
	print("Master problem solved:")

	if np.array_equal(rho_new[0,:,:],rho_initial) and abs(psi-lower_bound)<0.000001:
			reapeat_count = reapeat_count+1

	#update global lower bound
	lower_bound = psi
	if lower_bound==0:
			lower_bound = -1e-20
	rho = rho_new.copy()
	log_file = open(log_path,mode='a+')
	log_file.writelines(['iteration',str(i_gbd),'\nupper_bound:\t',str(upper_bound),'\tlower_bound:\t',\
		str(lower_bound),'\tsol_num:\t',str(1),'\tcut_num:\t',str(num_addcut),'\n'])
	log_file.writelines(['acc:',str(acc),'\tacc1:\t',str(acc1),'\tacc0:\t', str(acc0),'\n'])
	log_file.writelines(['total time for master problem:\t', str(time_total),'\n'])
	log_file.close()

	log_json.append({})
	log_json[i_gbd-1]['upper_bound'] = upper_bound
	log_json[i_gbd-1]['lower_bound'] = lower_bound
	log_json[i_gbd-1]['optimal_gap'] = (upper_bound-lower_bound)/abs(lower_bound)
	log_json[i_gbd-1]['total_time'] = time_total
	log_json[i_gbd-1]['total_cut'] = cut_total
		
	print("lower_bound",lower_bound) 
	print("upper_bound",upper_bound) 
	####################################################

	#from second iteration
	#solve #numsol primal problems and get #numsol cuts
	while ((upper_bound-lower_bound)/abs(lower_bound)>0.005) and (i_gbd<max_iter_num):
		i_gbd = i_gbd + 1

		feasible_flag_multi = np.zeros((numsol_real,1))
		p_D_multi = np.zeros((numsol_real,K0,L0))
		eta_multi = np.zeros((numsol_real,1))
		alpha_multi = np.zeros((numsol_real,1))
		lameda_multi = np.zeros((numsol_real,L0,1))
		miu_multi = np.zeros((numsol_real,K0,L0))
		nu_multi = np.zeros((numsol_real,K0,L0))
		theta_multi = np.zeros((numsol_real,L0,1))



		for i_sol in range(numsol_real):
			#solve the primal problem 
			feasible_flag, p_D, eta, lameda, miu, nu, theta = primal_solve(rho[i_sol,:,:])
			if feasible_flag==1:
				if upper_bound>-eta:
					rho_opt = rho[i_sol,:,:].copy()
					p_D_opt = np.array(p_D).copy()
					upper_bound = -eta
			else:
				#solve feasibilty check problem
				p_D, eta, alpha, lameda, miu, nu, theta = infeasible_solve(rho[i_sol,:,:])
				if alpha<1e-20:
					print("find a feasible point")
					feasible_flag=1
					if upper_bound>-eta:
						rho_opt = rho[i_sol,:,:].copy()
						p_D_opt = np.array(p_D).copy()
						upper_bound = -eta
				else:
					print("find an infeasible point")
			feasible_flag_multi[i_sol] = feasible_flag
			p_D_multi[i_sol,:,:] = np.array(p_D).copy()
			eta_multi[i_sol] = eta
			lameda_multi[i_sol,:,:] = np.array(lameda).copy()
			miu_multi[i_sol,:,:] = np.array(miu).copy()
			nu_multi[i_sol,:,:] = np.array(nu).copy()
			theta_multi[i_sol,:,:] = np.array(theta).copy()

		#solve the relaxed master problem 
		print("Master problem solving:")
		cut_label=master_class_train.master_sol(feasible_flag_multi, p_D_multi, eta_multi, alpha_multi, lameda_multi,\
			miu_multi, nu_multi, theta_multi, numsol, rho, threshold, lower_bound, rho[0])

		rho_new, psi, numsol_real_new, num_addcut, num_acc, num_acc1, num_acc0, this_num, num_1, num_0, time_new= master_class_test.master_sol(feasible_flag_multi, p_D_multi, eta_multi, \
			alpha_multi, lameda_multi, miu_multi, nu_multi, theta_multi, numsol, rho, lower_bound, rho[0],model,cut_label,scaler)


		acc = num_acc/this_num
		if num_1==0:
			acc1 = 'None'
		else:
			acc1 = num_acc1/num_1
		if num_0==0:
			acc0 = 'None'
		else:
			acc0 = num_acc0/num_0

		acc_total = acc_total + num_acc
		acc1_total = acc1_total + num_acc1
		acc0_total = acc0_total + num_acc0
		num_total = num_total + this_num
		num1_total = num1_total + num_1
		num0_total = num0_total + num_0

		time_total = time_total + time_new
		cut_total = cut_total + num_addcut

		print("Master problem solved!")

		if np.array_equal(rho_new[0,:,:],rho[0,:,:]) and abs(psi-lower_bound)<0.000001:
			reapeat_count = reapeat_count+1

		#update global lower bound
		lower_bound = psi
		if lower_bound==0:
			lower_bound = -1e-20
		rho = rho_new.copy()
		log_file = open(log_path,mode='a+')
		log_file.writelines(['iteration',str(i_gbd),'\nupper_bound:\t',str(upper_bound),'\tlower_bound:\t',\
			str(lower_bound),'\tsol_num:\t',str(numsol_real),'\tcut_num:\t',str(num_addcut),'\n'])
		log_file.writelines(['acc:',str(acc),'\tacc1:\t',str(acc1),'\tacc0:\t', str(acc0),'\n'])
		log_file.writelines(['total time for master problem:\t', str(time_total),'\n'])
		log_file.close()

		log_json.append({})
		log_json[i_gbd-1]['upper_bound'] = upper_bound
		log_json[i_gbd-1]['lower_bound'] = lower_bound
		log_json[i_gbd-1]['optimal_gap'] = (upper_bound-lower_bound)/abs(lower_bound)
		log_json[i_gbd-1]['total_time'] = time_total
		log_json[i_gbd-1]['total_cut'] = cut_total
		

		print("lower_bound",lower_bound) 
		print("upper_bound",upper_bound)

		numsol_real = numsol_real_new

		acc = acc_total/num_total
		if num1_total==0:
			acc1 = 'None'
		else:
			acc1 = acc1_total/num1_total
		if num_0==0:
			acc0 = 'None'
		else:
			acc0 = acc0_total/num0_total


		if reapeat_count == 2:
			if abs(upper_bound-lower_bound)<10:
				convergence_flag = 1
				res_file = open(res_path,mode='a+')
				res_file.writelines(['Problem loosely solved','\n'])
				res_file.writelines(['rho_opt:\n',str(rho_opt),'\np_D_opt:\n',str(p_D_opt),'\neta:\t',str(-upper_bound),'\titeration:\t',str(i_gbd),'\n'])
				res_file.writelines(['acc:',str(acc),'\tacc1:\t',str(acc1),'\tacc0:\t', str(acc0),'\n'])
				res_file.writelines(['\n'])
				res_file.close()
			else:
				convergence_flag = -1
				res_file = open(res_path,mode='a+')
				res_file.writelines(['One more iteration needed!\n'])
				res_file.writelines(['rho_opt:\n',str(rho_opt),'\np_D_opt:\n',str(p_D_opt),'\niteration:\t',str(i_gbd),'\n'])
				res_file.writelines(['upper_bound:\t',str(upper_bound),'\tlower_bound:\t',str(lower_bound),'\n'])
				res_file.writelines(['acc:',str(acc),'\tacc1:\t',str(acc1),'\tacc0:\t', str(acc0),'\n'])
				res_file.writelines(['\n'])
				res_file.close()
				log_file = open(log_path,mode='a+')
				log_file.writelines(['\nOne more iteration needed!\n'])
				log_file.close()
			json_output(log_json,log_json_path)
			master_class_test.json_output()
			acc_array = np.array([acc_total,num_total,acc1_total,num1_total,acc0_total,num0_total])
			return p_D_opt, rho_opt, rho, convergence_flag, i_gbd, acc_array

	if i_gbd<max_iter_num:
		convergence_flag = 1
		res_file = open(res_path,mode='a+')
		res_file.writelines(['Problem solved','\n'])
		res_file.writelines(['rho_opt:\n',str(rho_opt),'\np_D_opt:\n',str(p_D_opt),'\neta:\t',str(-upper_bound),'\titeration:\t',str(i_gbd),'\n'])
		res_file.writelines(['acc:',str(acc),'\tacc1:\t',str(acc1),'\tacc0:\t', str(acc0),'\n'])
		res_file.writelines(['\n'])
		res_file.close()
	else:
		convergence_flag = 0
		res_file = open(res_path,mode='a+')
		res_file.writelines(['Problem cannot converge','\n'])
		res_file.writelines(['rho_opt:\n',str(rho_opt),'\np_D_opt:\n',str(p_D_opt),'\neta:\t',str(-upper_bound),'\n'])
		res_file.writelines(['upper_bound:\t',str(upper_bound),'\tlower_bound:\t',str(lower_bound),'\n'])
		res_file.writelines(['acc:',str(acc_total/num_total),'\tacc1:\t',str(acc1_total/num1_total),'\tacc0:\t', str(acc0_total/num0_total),'\n'])
		res_file.writelines(['\n'])
		res_file.close()
		
	json_output(log_json,log_json_path)
	master_class_test.json_output()
	acc_array = np.array([acc_total,num_total,acc1_total,num1_total,acc0_total,num0_total])

	#output
	return p_D_opt, rho_opt, rho, convergence_flag, i_gbd, acc_array

def primal_solve(rho):
	print("Primal problem solving:\n")
	rho = matlab.double(rho.tolist()) 
	exit_flag, p_D, eta, lameda, miu, nu, theta = eng.primal_solve(K,L,R_min_C,P_max_D,P_max_C,h_CD,h_D,h_CB,h_DB,rho,nargout=7)
	if exit_flag <=0:
		feasible_flag = 0
		print("Primal problem is infeasible.")
	else:
		feasible_flag = 1
		print("Primal problem solved.")
	return feasible_flag, p_D, eta, lameda, miu, nu, theta


def infeasible_solve(rho):
	print("Feasiblity-check problem solving:\n")
	rho = matlab.double(rho.tolist()) 
	p_D, eta, alpha, lameda, miu, nu, theta = eng.infeasible_solve3(K,L,R_min_C,P_max_D,P_max_C,h_CD,h_D,h_CB,h_DB,rho,nargout=7)
	return p_D, eta, alpha, lameda, miu, nu, theta

def data_extract(mattr):
	data= sio.loadmat(mattr)
	K = data['K'][0,0]
	L = data['L'][0,0]
	R_min_C = data['R_min_C'][0,0]
	P_max_D = data['P_max_D'][0,0]
	P_max_C = data['P_max_C'][0,0]
	H_CD = np.transpose(data['H_CD'])
	H_D = np.transpose(data['H_D'])
	H_CB = np.transpose(data['H_CB'])
	H_DB = np.transpose(data['H_DB'])

	return K,L,R_min_C,P_max_D,P_max_C,H_CD,H_D,H_CB,H_DB

def para():
	a, b, p_max = eng.para(K,L,R_min_C,P_max_D,P_max_C,h_CD,h_D,h_CB,h_DB,nargout=3)
	a = np.array(a)
	b = np.array(b)
	p_max = np.array(p_max)
	return a, b, p_max


def json_output(log_json,log_json_path):
	f=open(log_json_path,'w',encoding='utf-8')
	json.dump(log_json,f,indent=4,ensure_ascii=False)










