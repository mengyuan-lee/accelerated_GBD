# -*- coding:utf-8 -*-
##solving master problem for multi-cut GBD
import cplex
from cplex.exceptions import CplexError
from cplex.exceptions import CplexSolverError
import sys
import numpy as np
import copy
import math
import time


class Master_class:
    my_prob = cplex.Cplex()
    init_flag = False

    def __init__(self, p_K, p_L, p_h_CD, p_h_CB, p_h_D, p_h_DB, p_P_max_D, p_R_min_C, p_P_max_C, p_p_max_k_l, p_a, p_b):
        self.my_prob = cplex.Cplex()
        self.init_flag == False

        self.my_prob.objective.set_sense(self.my_prob.objective.sense.minimize)
        global K, L, h_CD, h_CB, h_D, h_DB, P_max_D, R_min_C, P_max_C, p_max_k_l, a, b

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

        #print("my_row:",my_new_row)        
        #print("my_sense:",my_sense)
        #print("rhs:",my_new_rhs)
        self.my_prob.linear_constraints.add(
            lin_expr=my_new_row, senses=my_sense, rhs=my_new_rhs)

    def lplex1(self, my_obj, my_lb, my_ub, my_colnames, my_new_row, my_new_rhs, K, L,numsol, ret_flag):
        
        try:
            handle = self.populatebyrow(
                my_obj, my_lb, my_ub, my_colnames, my_new_row, my_new_rhs, K, L)
        except CplexSolverError:
            print("Exception raised during populate")
            return

        if ret_flag == True:
            print("test_numsol",numsol)
            self.my_prob.parameters.mip.pool.capacity.set(numsol)
            self.my_prob.parameters.mip.pool.replace.set(1)
            time_start =time.clock()
            self.my_prob.populate_solution_pool()
            time_use = time.clock() 


            num_solution = self.my_prob.solution.pool.get_num()
            print("The solution pool contains %d solutions." % num_solution)
            # meanobjval = cpx.solution.pool.get_mean_objective_value()

            numsol_real = min(numsol, num_solution)
            sol_pool = []
        
            obj_temp = np.zeros(numsol_real)
            for i in range(numsol_real):
                obj_temp[i] = self.my_prob.solution.pool.get_objective_value(i) 
            new_index = sorted(range(len(obj_temp)), key=lambda k: obj_temp[k])
            print(obj_temp)
            print(new_index)

            for j in range(numsol_real):
                i = new_index[j]
                objval_i = self.my_prob.solution.pool.get_objective_value(i)
                x_i = self.my_prob.solution.pool.get_values(i)
                nb_vars=len(x_i)
                sol = []
                for k in range(nb_vars):
                    sol.append(x_i[k])
                sol_pool.append(sol)
                print("object:",i,objval_i)
                print("value:",i,x_i)

            print("pools =",sol_pool)

            # Print information about the incumbent
            print("Objective value of the incumbent  = ",
                self.my_prob.solution.get_objective_value())

            r_Psi = self.my_prob.solution.get_objective_value()

            r_rho = np.ones([numsol_real,K, L], dtype=int)
            for i in range(numsol_real):
                x_i = sol_pool[i]
                index_i = 0
                index_j = -1
                for index in range(0, K*L):
                    index_j = index_j+1
                    if index_j >= L:
                        index_j = 0
                        index_i = index_i + 1
                    r_rho[i,index_i,index_j] = x_i[index]
            print("r_rho",r_rho)
            print("r_psi",r_Psi)


            return r_rho, r_Psi, numsol_real, time_use-time_start
        else:
            return -1, -1, -1, -1

    def master_sol(self,feasible_flag_multi,p_D_multi,eta_multi,alpha_multi,lameda_multi,miu_multi,nu_multi,theta_multi,numsol):
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
        # print(feasible_flag_multi)
        # print("test")
        # print(this_num)
        for num_index in range(this_num):
            feasible_flag = feasible_flag_multi[num_index]
            p_D = p_D_multi[num_index]
            eta = eta_multi[num_index]
            alpha = alpha_multi[num_index]
            lameda = lameda_multi[num_index]
            miu = miu_multi[num_index]
            nu = nu_multi[num_index]
            theta = theta_multi[num_index]

            # feasible_flag_multi,p_D_multi,eta_multi,alpha_multi,lameda_multi,miu_multi,nu_multi,theta_multi,numsol

            if feasible_flag:
                variables_names = copy.deepcopy(my_colnames)
                variables_coef = np.zeros(K*L+1, dtype=np.float_)
                rho_coef = np.zeros([K, L], dtype=np.float_)
                new_rhs = 0.0

                # ψ 
                variables_coef[K*L] = 1.0

                # constant
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
                        # print(index_k,index_l,variables_names[index_temp])

                new_row = [variables_names, variables_coef.tolist()]

                new_rhs=new_rhs.tolist()

                #print("feasible")
                #print(my_obj,my_lb,my_ub,my_colnames,new_row,new_rhs,K,L,numsol,num_index,num_index==this_num-1)
                r_rho, r_Psi, r_solnum, time_use = self.lplex1(
                    my_obj, my_lb, my_ub, my_colnames, new_row, new_rhs, K, L, numsol, num_index==this_num-1)

                #res_file = open('result.txt',mode='a+')
                #res_file.writelines(['coef:\t',str(variables_coef),'\nflag:\t',str(num_index==this_num-1),'\ncut_num_sofar\t',str(self.my_prob.linear_constraints.get_num()),'\n'])
                #res_file.writelines(['\n'])
                #res_file.close()

            else:
                variables_names = copy.deepcopy(my_colnames)
                variables_coef = np.zeros(K*L+1, dtype=np.float_)
                rho_coef = np.zeros([K, L], dtype=np.float_)
                new_rhs = 0.0

                # constant
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

                print("infeasible")
                r_rho, r_Psi, r_solnum, time_use = self.lplex1(
                    my_obj, my_lb, my_ub, my_colnames, new_row, new_rhs, K, L,numsol,num_index==this_num-1)
        return r_rho, r_Psi, r_solnum, time_use   

        