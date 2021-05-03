# -*- coding:utf-8 -*-
#solving master problem for classical GBD
import cplex
from cplex.exceptions import CplexError
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
            # print(my_ctype)

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
                str = "rho"+repr(index_i)+"_"+repr(index_j)
                my_colnames.append(str)

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

    def lplex1(self, my_obj, my_lb, my_ub, my_colnames, my_new_row, my_new_rhs, K, L):
        try:
            handle = self.populatebyrow(
                my_obj, my_lb, my_ub, my_colnames, my_new_row, my_new_rhs, K, L)
            #self.my_prob.parameters.mip.pool.capacity.set(numsol)
            #self.my_prob.parameters.mip.pool.relgap.set(0.1)
            self.my_prob.parameters.mip.pool.replace.set(1)
            time_start =time.clock()
            self.my_prob.populate_solution_pool()
            time_use = time.clock() 

        except CplexError as exc:
            print(exc)
            print("CplexError")
            return -1,-1,-1

        print()
        # solution.get_status() returns an integer code
        # print("Solution status = ", self.my_prob.solution.get_status(), ":", end=' ')
        # print(self.my_prob.solution.status[self.my_prob.solution.get_status()])
        print("Solution value  = ", self.my_prob.solution.get_objective_value())

        numcols = self.my_prob.variables.get_num()
        numrows = self.my_prob.linear_constraints.get_num()

        x = self.my_prob.solution.get_values()

        '''
        for j in range(numcols):
            print("Column %d:  Value = %10f" % (j, x[j]))
        '''
        j = numcols-1

        r_Psi = x[j]
        r_rho = np.ones([K, L], dtype=int)
        index_i = 0
        index_j = -1
        for index in range(0, K*L):
            index_j = index_j+1
            if index_j >= L:
                index_j = 0
                index_i = index_i + 1
            r_rho[index_i][index_j] = x[index]
        print(r_rho, r_Psi)

        return r_rho, r_Psi, time_use-time_start


    def master_sol(self, feasible_flag, p_D, eta, alpha, lameda, miu, nu,theta):
        #print(K, L)
        lameda=np.array(lameda)
        miu=np.array(miu)
        nu=np.array(nu)
        p_D=np.array(p_D)
        theta=np.array(theta)

        
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
            str = "rho"+repr(index_i)+"_"+repr(index_j)
            my_colnames.append(str)

        my_colnames.append("Psi")

        if feasible_flag:
            variables_names = copy.deepcopy(my_colnames)
            variables_coef = np.zeros(K*L+1, dtype=np.float_)
            rho_coef = np.zeros([K, L], dtype=np.float_)
            new_rhs = 0.0

            # ψ 
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

            print("feasible")
            r_rho, r_Psi, time_use = self.lplex1(
                my_obj, my_lb, my_ub, my_colnames, new_row, new_rhs, K, L)
            return r_rho, r_Psi, time_use

        else:
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

            print("infeasible")
            r_rho, r_Psi, time_use = self.lplex1(
                my_obj, my_lb, my_ub, my_colnames, new_row, new_rhs, K, L)
            return r_rho, r_Psi, time_use
        return -1
        