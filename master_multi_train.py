# -*- coding:utf-8 -*-
#solving master problem for multi-cut GBD and collect features & labels
import cplex
from cplex.exceptions import CplexError
from cplex.exceptions import CplexSolverError
import sys
import numpy as np
import copy
import math
import os
import json
import time


output_num = 0

class Master_class:
    my_prob = cplex.Cplex()
    init_flag = False

    def __init__(self, p_K, p_L, p_h_CD, p_h_CB, p_h_D, p_h_DB, p_P_max_D, p_R_min_C, p_P_max_C, p_p_max_k_l, p_a, p_b, p_i):
        self.my_prob = cplex.Cplex()
        self.init_flag == False

        self.my_prob.objective.set_sense(self.my_prob.objective.sense.minimize)
        global K, L, h_CD, h_CB, h_D, h_DB, P_max_D, R_min_C, P_max_C, p_max_k_l, a, b, output_num

        K = p_K
        L = p_L
        h_CD = p_h_CD
        h_CB = p_h_CB
        h_D = p_h_D
        h_DB = p_h_DB
        P_max_D = p_P_max_D
        R_min_C = p_R_min_C
        P_max_C = p_P_max_C
        p_max_k_l = p_p_max_k_l
        a = p_a
        b = p_b 
        output_num = p_i+1

        self.features = []
        self.labels = []        
        self.numindex = 0
        self.i_gbd = 0
        self.rho_dict = {}

        

    def populatebyrow(self, my_obj, my_lb, my_ub, my_colnames, my_new_row, my_new_rhs, K, L):

        if self.init_flag == False:
            self.init_flag = True
            my_ctype = ""
            for index in range(0, K*L):
                my_ctype = my_ctype + "I"
            my_ctype = my_ctype + "C"

            self.my_prob.variables.add(
                obj=my_obj, lb=my_lb, ub=my_ub, names=my_colnames, types=my_ctype)
            
            my_colnames = []

            index_i = 1
            index_j = 0
            for index in range(0, K*L):
                index_j = index_j+1
                if index_j > L:
                    index_j = 1
                    index_i = index_i + 1
                str_temp = "rho"+repr(index_i)+"_"+repr(index_j)
                my_colnames.append(str_temp)

            my_colnames.append("Psi")
            t_sense = "L"
            t_new_rhs = [1]
            for index_k in range(0,K):
                variables_coef = np.zeros(K*L+1, dtype=np.float_)
                for index_l in range(0,L):
                    variables_coef[index_k*L+index_l]=1.0
                
                t_new_row = [my_colnames, variables_coef.tolist()]
                t_new_row = [t_new_row]   
                
                #print(t_new_row,t_sense,t_new_rhs)
                self.my_prob.linear_constraints.add(
                        lin_expr=t_new_row, senses=t_sense, rhs=t_new_rhs)

        my_new_row = [my_new_row]
        my_sense = "G"

        self.my_prob.linear_constraints.add(
            lin_expr=my_new_row, senses=my_sense, rhs=my_new_rhs)

    def lplex1(self, my_obj, my_lb, my_ub, my_colnames, my_new_row, my_new_rhs, K, L,numsol, ret_flag, last_obj_value):
        try:
            handle = self.populatebyrow(
                my_obj, my_lb, my_ub, my_colnames, my_new_row, my_new_rhs, K, L)
        except CplexSolverError:
            print("Exception raised during populate")
            return

        self.my_prob.solve()
        self.label[-1]['CI'] = self.my_prob.solution.get_objective_value() -last_obj_value

        if ret_flag == True:
            for i in range(0,self.numindex):
                self.label[i]['CT'] = self.my_prob.solution.get_objective_value() - last_obj_value
                if self.label[i]['CT'] ==0:
                    self.label[i]['VALUABLE'] = 0
                else:
                    if i==0:
                        self.label[i]['VALUABLE'] = 1
                    else:
                        if self.label[i]['CI']/self.label[i]['CT']> self.threshold and self.label[i-1]['CI']/self.label[i-1]['CT']> self.threshold:
                            self.label[i]['VALUABLE'] = 0
                        else:
                            self.label[i]['VALUABLE'] = 1


    def json_output(self):
        output = {'features':self.features,'labels':self.labels}
        print("feature:",output)
        if not os.path.exists('outputs/problem_'+str(K)+'_'+str(L)):
            os.makedirs('outputs/problem_'+str(K)+'_'+str(L))
        output_file_name = 'outputs/problem_'+str(K)+'_'+str(L)+'/data'+ str(output_num) + '.json'
        f=open(output_file_name,'w',encoding='utf-8')
        json.dump(output,f,indent=4,ensure_ascii=False)

    def master_sol(self,feasible_flag_multi,p_D_multi,eta_multi,alpha_multi,lameda_multi,miu_multi,nu_multi,theta_multi,numsol,rho_multi,threshold, last_obj_value, last_rho):
        self.i_gbd = self.i_gbd + 1
        self.threshold = threshold
        
        N = K*L + 1

        my_obj = np.zeros(K*L+1, dtype=np.float_)
        my_obj[K*L] = 1.0
        # print(my_obj)
        my_lb = np.zeros(K*L+1, dtype=np.float_)
        my_ub = np.ones(K*L+1, dtype=np.float_)

        my_lb[K*L] = -cplex.infinity
        my_ub[K*L] = cplex.infinity

        my_colnames = []

        index_i = 1
        index_j = 0
        for index in range(0, K*L):
            index_j = index_j+1
            if index_j > L:
                index_j = 1
                index_i = index_i + 1
            str_temp = "rho"+repr(index_i)+"_"+repr(index_j)
            my_colnames.append(str_temp)

        my_colnames.append("Psi")

        this_num = feasible_flag_multi.shape[0]
        self.feature = []
        self.label = []

        self.numindex = this_num


        for num_index in range(this_num):

            self.feature.append({})

            self.label.append({})      

            self.feature[num_index]['index'] = num_index+1 


            rho_tuple = copy.deepcopy(rho_multi[num_index])
            rho_tuple = rho_tuple.flatten()
            rho_tuple = tuple(rho_tuple)
            print(rho_tuple)
            print(type(rho_tuple))

            if rho_tuple in self.rho_dict:
                self.rho_dict[rho_tuple] = self.rho_dict[rho_tuple] + 1
                self.feature[num_index]['num_cut'] = self.rho_dict[rho_tuple]
            else:
                self.feature[num_index]['num_cut'] = 1
                self.rho_dict[rho_tuple] = 1

     

            feasible_flag = feasible_flag_multi[num_index]

        
            p_D = p_D_multi[num_index]

            eta = eta_multi[num_index]
            alpha = alpha_multi[num_index]
            lameda = lameda_multi[num_index]
            miu = miu_multi[num_index]
            nu = nu_multi[num_index]
            theta = theta_multi[num_index]

            self.feature[num_index]['i_gbd']=self.i_gbd

            if feasible_flag:
                self.feature[num_index]['cut_type']=1
                variables_names = copy.deepcopy(my_colnames)
                variables_coef = np.zeros(K*L+1, dtype=np.float_)
                rho_coef = np.zeros([K, L], dtype=np.float_)
                new_rhs = 0.0

                variables_coef[K*L] = 1.0

                new_rhs -= eta

                for index_l in range(0, L):
                    new_rhs -= lameda[index_l]*P_max_D
                
                for index_k in range(0, K):
                    for index_l in range(0, L):
                        new_rhs += lameda[index_l]*p_D[index_k][index_l]

                for index_k in range(0, K):
                    for index_l in range(0, L):
                        new_rhs += miu[index_k][index_l] * \
                            (p_D[index_k][index_l])

                for index_k in range(0, K):
                    for index_l in range(0,L):
                        new_rhs -= nu[index_k][index_l]*p_D[index_k][index_l]

                for index_l in range(0, L):
                    new_rhs += theta[index_l]*eta

                for index_k in range(0, K):
                    for index_l in range(0, L):
                        new_rhs -= theta[index_l]*math.log(1+p_D[index_k][index_l]/(
                            a[index_k][index_l]+b[index_k][index_l]*p_D[index_k][index_l]), 2.0)
                

                #print("rho_coef0:",rho_coef)
                for index_l in range(0, L):
                    for index_k in range(0, K):
                        rho_coef[index_k][index_l] = miu[index_k][index_l] * \
                            p_max_k_l[index_k][index_l]

                for index_k in range(0, K):
                    for index_l in range(0, L):
                        index_temp = index_k*L + index_l
                        variables_coef[index_temp] = rho_coef[index_k][index_l]

                new_row = [variables_names, variables_coef.tolist()]

                new_rhs=new_rhs.tolist()
                
                L_value = 0
                L_value = L_value - eta

                for index_l in range(0, L):
                    L_value -= lameda[index_l]*P_max_D

                for index_k in range(0, K):
                    for index_l in range(0, L):
                        L_value += lameda[index_l]*p_D[index_k][index_l]


                for index_k in range(0, K):
                    for index_l in range(0, L):
                        L_value += miu[index_k][index_l] * \
                            (p_D[index_k][index_l]-last_rho[index_k][index_l]*p_max_k_l[index_k][index_l])


                for index_k in range(0, K):
                    for index_l in range(0,L):
                        L_value -= nu[index_k][index_l]*p_D[index_k][index_l]

                for index_l in range(0, L):
                    L_value += theta[index_l]*eta

                for index_k in range(0, K):
                    for index_l in range(0, L):
                        L_value -= theta[index_l]*math.log(1+p_D[index_k][index_l]/(
                            a[index_k][index_l]+b[index_k][index_l]*p_D[index_k][index_l]), 2.0)

                
                self.feature[num_index]['cut_violation']=L_value[0]-last_obj_value
                
                self.lplex1(my_obj, my_lb, my_ub, my_colnames, new_row, new_rhs, K, L, numsol,num_index==this_num-1,last_obj_value)

            else:
                self.feature[num_index]['cut_type']=0

                variables_names = copy.deepcopy(my_colnames)
                variables_coef = np.zeros(K*L+1, dtype=np.float_)
                rho_coef = np.zeros([K, L], dtype=np.float_)
                new_rhs = 0.0


                for index_l in range(0, L):
                    new_rhs -= lameda[index_l]*(P_max_D+alpha)
                
                for index_k in range(0, K):
                    for index_l in range(0, L):
                        new_rhs += lameda[index_l]*p_D[index_k][index_l]

                for index_k in range(0, K):
                    for index_l in range(0, L):
                        new_rhs += miu[index_k][index_l] * \
                            (p_D[index_k][index_l]-alpha)

                for index_k in range(0, K):
                    for index_l in range(0,L):
                        new_rhs -= nu[index_k][index_l]*(p_D[index_k][index_l]+alpha)

                for index_l in range(0, L):
                    new_rhs += theta[index_l]*(eta-alpha)
                
                for index_k in range(0, K):
                    for index_l in range(0, L):
                        new_rhs -= theta[index_l]*math.log(1+p_D[index_k][index_l]/(
                            a[index_k][index_l]+b[index_k][index_l]*p_D[index_k][index_l]), 2.0)

                for index_l in range(0, L):
                    for index_k in range(0, K):
                        rho_coef[index_k][index_l] = miu[index_k][index_l] * \
                            p_max_k_l[index_k][index_l]

                for index_k in range(0, K):
                    for index_l in range(0, L):
                        index_temp = index_k*L + index_l
                        variables_coef[index_temp] = rho_coef[index_k][index_l]
                        # print(index_k,index_l,variables_names[index_temp])

                new_row = [variables_names, variables_coef.tolist()]

                L_value = 0
                
                for index_l in range(0, L):
                    L_value -= lameda[index_l]*(P_max_D+alpha)
                
                for index_k in range(0, K):
                    for index_l in range(0, L):
                        L_value += lameda[index_l]*p_D[index_k][index_l]

                for index_k in range(0, K):
                    for index_l in range(0, L):
                        L_value += miu[index_k][index_l] * \
                            (p_D[index_k][index_l]-last_rho[index_k][index_l]-alpha)

                for index_k in range(0, K):
                    for index_l in range(0,L):
                        L_value -= nu[index_k][index_l]*(p_D[index_k][index_l]+alpha)

                for index_l in range(0, L):
                    L_value += theta[index_l]*(eta-alpha)
                
                for index_k in range(0, K):
                    for index_l in range(0, L):
                        L_value -= theta[index_l]*math.log(1+p_D[index_k][index_l]/(
                            a[index_k][index_l]+b[index_k][index_l]*p_D[index_k][index_l]), 2.0)

                self.feature[num_index]['cut_violation']=L_value[0]

                print("infeasible")
                self.lplex1(my_obj, my_lb, my_ub, my_colnames, new_row, new_rhs, K, L,numsol,num_index==this_num-1,last_obj_value)
                   
        self.features.extend(self.feature)
        self.labels.extend(self.label)

        return self.label

        