# -*- coding:utf-8 -*-
#solving master problem for calssical GBD and collect features & labels
import cplex
from cplex.exceptions import CplexError
from cplex.exceptions import CplexSolverError
import sys
import numpy as np
import copy
import math

import json
import os

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
        output_num = p_i + 1

        self.last_objective_value = -sys.maxsize

        self.features = []
        self.labels = []        
        self.i_gbd = 0
        self.rho_dict = {}

        self.last_rho = np.zeros([K, L], dtype=int)
        self.last_rho_index = 0
        

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

    def lplex1(self, my_obj, my_lb, my_ub, my_colnames, my_new_row, my_new_rhs, K, L,numsol):
        try:
            handle = self.populatebyrow(
                my_obj, my_lb, my_ub, my_colnames, my_new_row, my_new_rhs, K, L)
        except CplexSolverError:
            print("Exception raised during populate")
            return

        print("test_numsol",numsol)
        self.my_prob.parameters.mip.pool.capacity.set(numsol)
        self.my_prob.parameters.mip.pool.replace.set(1)
        self.my_prob.populate_solution_pool()
        

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

        
        self.label[-1]['CI'] = self.my_prob.solution.get_objective_value() - self.last_objective_value
                
        self.last_objective_value = self.my_prob.solution.get_objective_value()

        r_Psi = self.my_prob.solution.get_objective_value()


            
        r_rho = np.zeros((K,L))  
        i = np.random.randint(0,numsol_real)
        x_i = self.my_prob.solution.pool.get_values(i)
        index_i = 0
        index_j = -1
        for index in range(0, K*L):
            index_j = index_j+1
            if index_j >= L:
                index_j = 0
                index_i = index_i + 1
            r_rho[index_i,index_j] = x_i[index]

        self.last_rho = r_rho.copy()
        self.last_rho_index = i


        print("r_rho",r_rho)
        print("r_psi",r_Psi)
        return r_rho, r_Psi

    def json_output(self):
        for i in range(self.i_gbd-1):
            if self.labels[i]['CI']>self.threshold*self.labels[i+1]['CI']:
                self.labels[i]['VALUABLE'] = 1
            else:
                self.labels[i]['VALUABLE'] = 0
        self.labels[self.i_gbd-1]['VALUABLE'] = 1
        output = {'features':self.features,'labels':self.labels}
        if not os.path.exists('outputs/problem_'+str(K)+'_'+str(L)):
            os.makedirs('outputs/problem_'+str(K)+'_'+str(L))
        output_file_name = 'outputs/problem_'+str(K)+'_'+str(L)+'/data'+ str(output_num) + '.json'
        f=open(output_file_name,'w',encoding='utf-8')
        json.dump(output,f,indent=4,ensure_ascii=False)

    def master_sol(self,feasible_flag,p_D,eta,alpha,lameda,miu,nu,theta,numsol,rho,threshold):
        self.i_gbd = self.i_gbd + 1
        self.threshold = threshold

        lameda=np.array(lameda)
        miu=np.array(miu)
        nu=np.array(nu)
        p_D=np.array(p_D)
        theta=np.array(theta)
        rho = np.array(rho)
        
        N = K*L + 1

        my_obj = np.zeros(K*L+1, dtype=np.float_)
        my_obj[K*L] = 1.0

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

        # print(feasible_flag_multi)
        # print("test")
        # print(this_num)
        self.feature = []
        self.label = []

        row_temp = []
        rhs_temp = [] 
        # new_row, new_rhs, 

        self.feature.append({})

        self.label.append({})       

        # self.feature[i]['num_cut'] = 
        rho_tuple = copy.deepcopy(rho)
        rho_tuple = rho_tuple.flatten()
        rho_tuple = tuple(rho_tuple)
        print(rho_tuple)
        print(type(rho_tuple))

        if rho_tuple in self.rho_dict:
            self.rho_dict[rho_tuple] = self.rho_dict[rho_tuple] + 1
            self.feature[0]['num_cut'] = self.rho_dict[rho_tuple]
        else:
            self.feature[0]['num_cut'] = 1
            self.rho_dict[rho_tuple] = 1

        # feasible_flag_multi,p_D_multi,eta_multi,alpha_multi,lameda_multi,miu_multi,nu_multi,theta_multi,numsol
        self.feature[0]['i_gbd']=self.i_gbd

        if feasible_flag:
            self.feature[0]['cut_type']=1
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

            print("feasible")
            print(my_obj,my_lb,my_ub,my_colnames,new_row,new_rhs,K,L,numsol)
            
            L_value = 0
            L_value = L_value - eta

            for index_l in range(0, L):
                L_value -= lameda[index_l]*P_max_D
                
            # print("test1",type(L_value))

            for index_k in range(0, K):
                for index_l in range(0, L):
                    L_value += lameda[index_l]*p_D[index_k][index_l]

            # print("test2",type(L_value))

            for index_k in range(0, K):
                for index_l in range(0, L):
                    L_value += miu[index_k][index_l] * \
                        (p_D[index_k][index_l]-self.last_rho[index_k][index_l]*p_max_k_l[index_k][index_l])

            # print("test3",type(L_value))


            for index_k in range(0, K):
                for index_l in range(0,L):
                    L_value -= nu[index_k][index_l]*p_D[index_k][index_l]

            # print("test4",type(L_value))


            for index_l in range(0, L):
                L_value += theta[index_l]*eta

            for index_k in range(0, K):
                for index_l in range(0, L):
                    L_value -= theta[index_l]*math.log(1+p_D[index_k][index_l]/(
                        a[index_k][index_l]+b[index_k][index_l]*p_D[index_k][index_l]), 2.0)

                
            self.feature[0]['cut_violation']=L_value[0]-self.last_objective_value
            self.feature[0]['cut_order']=self.last_rho_index

                
            row_temp.append(new_row)
            rhs_temp.append(new_rhs)
            r_rho, r_Psi= self.lplex1(
                my_obj, my_lb, my_ub, my_colnames, new_row, new_rhs, K, L, numsol)

        else:
            self.feature[0]['cut_type']=0

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
                        (p_D[index_k][index_l]-self.last_rho[index_k][index_l]-alpha)

            for index_k in range(0, K):
                for index_l in range(0,L):
                    L_value -= nu[index_k][index_l]*(p_D[index_k][index_l]+alpha)

            for index_l in range(0, L):
                L_value += theta[index_l]*(eta-alpha)
                
            for index_k in range(0, K):
                for index_l in range(0, L):
                    L_value -= theta[index_l]*math.log(1+p_D[index_k][index_l]/(
                        a[index_k][index_l]+b[index_k][index_l]*p_D[index_k][index_l]), 2.0)

            self.feature[0]['cut_violation']=L_value[0]
            self.feature[0]['cut_order']=self.last_rho_index

            print("infeasible")

            row_temp.append(new_row)
            rhs_temp.append(new_rhs)

            r_rho, r_Psi= self.lplex1(
                my_obj, my_lb, my_ub, my_colnames, new_row, new_rhs, K, L,numsol)

                   
        self.features.extend(self.feature)
        self.labels.extend(self.label)
        return r_rho, r_Psi

        