# -*- coding:utf-8 -*-
#collect data using calssical GBD
import sys
import numpy as np
import matlab
import matlab.engine
from master_single_train import Master_class
import scipy.io as sio 
import os
import timeit

eng = matlab.engine.start_matlab()

#global constant 
#K, L, h_CD, h_CB, h_D, h_DB, P_max_D, R_min_C, P_max_C


def gbd_single_train(K0,L0,R_min_C0,P_max_D0,P_max_C0,h_CD0,h_D0,h_CB0,h_DB0,rho_initial,numsol,threshold,res_path,log_path,i):
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
	

	#set iteration index
	i_gbd = 0
	#set maximum iteration num 
	max_iter_num = 1E5
	#select an initial value for rho which makes the primal problem feasible
	rho = rho_initial.copy()

	#set global upper bound and lower bound
	upper_bound = sys.maxsize
	lower_bound = -sys.maxsize
	#master calss initialization
	master_class = Master_class(K0, L0, h_CD0, h_CB0, h_D0, h_DB0, P_max_D0, R_min_C0, P_max_C0, p_max, a, b, i)

	
	alpha = 0

	#avoid local minimum
	reapeat_count = 0
	#optimal rho_result
	rho_opt = np.zeros((K0,L0))
	p_D_opt = np.zeros((K0,L0))

	

	while ((upper_bound-lower_bound)/abs(lower_bound)>0.005) and (i_gbd<max_iter_num):
		i_gbd = i_gbd + 1

		#solve the primal problem 
		feasible_flag, p_D, eta, lameda, miu, nu, theta = primal_solve(rho)
		if feasible_flag==True:
			if upper_bound>-eta:
				rho_opt = rho.copy()
				p_D_opt = np.array(p_D).copy()
				upper_bound = -eta
		else:
			#solve feasibilty check problem 
			p_D, eta, alpha, lameda, miu, nu, theta = infeasible_solve(rho)
			if alpha<1e-20:
				print("find a feasible point")
				feasible_flag=True
				if upper_bound>-eta:
					rho_opt = rho.copy()
					p_D_opt = np.array(p_D).copy()
					upper_bound = -eta
			else:
				print("find an infeasible point")

		#solve the relaxed master problem 
		print("Master problem solving:")
		rho_new, psi = master_class.master_sol(feasible_flag, p_D, eta, alpha, lameda, miu, nu, theta, numsol, rho, threshold)
		print("Master problem solved!")

		if np.array_equal(rho_new,rho) and abs(psi-lower_bound)<0.000001:
			reapeat_count = reapeat_count+1

		#update global lower bound
		lower_bound = psi
		if lower_bound==0:
			lower_bound = -1e-20
		rho = rho_new.copy()
		log_file = open(log_path,mode='a+')
		log_file.writelines(['iteration',str(i_gbd),'\nupper_bound:\t',str(upper_bound),'\tlower_bound:\t',str(lower_bound),'\n'])
		log_file.close()
		

		print("lower_bound",lower_bound) 
		print("upper_bound",upper_bound) 

		if reapeat_count == 2:
			if abs(upper_bound-lower_bound)<10:
				convergence_flag = 1
				res_file = open(res_path,mode='a+')
				res_file.writelines(['Problem loosely solved','\n'])
				res_file.writelines(['rho_opt:\n',str(rho_opt),'\np_D_opt:\n',str(p_D_opt),'\neta:\t',str(-upper_bound),'\titeration:\t',str(i_gbd),'\n'])
				res_file.writelines(['\n'])
				res_file.close()
			else:
				convergence_flag = -1
				res_file = open(res_path,mode='a+')
				res_file.writelines(['One more iteration needed!\n'])
				res_file.writelines(['rho_opt:\n',str(rho_opt),'\np_D_opt:\n',str(p_D_opt),'\niteration:\t',str(i_gbd),'\n'])
				res_file.writelines(['upper_bound:\t',str(upper_bound),'\tlower_bound:\t',str(lower_bound),'\n'])
				res_file.writelines(['\n'])
				res_file.close()
				log_file = open(log_path,mode='a+')
				log_file.writelines(['\nOne more iteration needed!\n'])
				log_file.close()
			master_class.json_output()
			return p_D_opt, rho_opt, rho, convergence_flag, i_gbd

	if i_gbd<max_iter_num:
		convergence_flag = 1
		res_file = open(res_path,mode='a+')
		res_file.writelines(['Problem solved','\n'])
		res_file.writelines(['rho_opt:\n',str(rho_opt),'\np_D_opt:\n',str(p_D_opt),'\neta:\t',str(-upper_bound),'\titeration:\t',str(i_gbd),'\n'])
		res_file.writelines(['\n'])
		res_file.close()
	else:
		convergence_flag = 0
		res_file = open(res_path,mode='a+')
		res_file.writelines(['Problem cannot converge','\n'])
		res_file.writelines(['rho_opt:\n',str(rho_opt),'\np_D_opt:\n',str(p_D_opt),'\neta:\t',str(-upper_bound),'\n'])
		res_file.writelines(['upper_bound:\t',str(upper_bound),'\tlower_bound:\t',str(lower_bound),'\n'])
		res_file.writelines(['\n'])
		res_file.close()

	master_class.json_output()

	#output
	return p_D_opt, rho_opt, rho, convergence_flag, i_gbd


def primal_solve(rho):
	print("Primal problem solving:\n")
	rho = matlab.double(rho.tolist()) 
	exit_flag, p_D, eta, lameda, miu, nu, theta = eng.primal_solve(K,L,R_min_C,P_max_D,P_max_C,h_CD,h_D,h_CB,h_DB,rho,nargout=7)
	if exit_flag <=0:
		feasible_flag = False
		print("Primal problem is infeasible.")
	else:
		feasible_flag = True
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
	





