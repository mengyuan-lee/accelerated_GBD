function [exit_flag,p,eta,lameda,miu,nu,theta] = primal_solve(K,L,R_min_C,P_max_D,P_max_C,h_CD,h_D,h_CB,h_DB,rho)
%output: exit_flag: Reason fmincon stopped, returned as an integer.
%        p eta
%        lameda,miu,theta: Lagrangian multiplier
%input:  K number of CUs
%        L number of D2D links
%        R_min_C: minimum rate of CUs
%        P_max_D: maximum d2d linnk power
%        P_max_C: maximum CUs
%        h_CD:channel gain between CU and the receiver of D2D pair 
%        h_D:channel gain of D2D pair 
%        h_CB:channel gain between CU and base station
%        h_CB:channel gain between the transmitter of D2D pair and base station
%        rho_d:channel allocation K*L


[K,L,a,b,P_max_D,p_max]=const(K,L,R_min_C,P_max_D,P_max_C,h_CD,h_D,h_CB,h_DB,rho);


%objective function
fun = @(x)-x(1);
%initialization
x0 = zeros(K*L+1,1);
ub = ones(K*L+1,1);
ub(1) = inf;
ub(2:K*L+1) = rho(:).*p_max(:);
%-----------------constraints---------------------
lb = zeros(K*L+1,1);
%linear inequality 
n_v = length(x0);
A = zeros(L,n_v);
B = zeros(L,1);

%(10a)
for i = 1:L
    for j = 1:K
        A(i,1+(i-1)*K+j)= 1;
    end
    B(i,1) = P_max_D;
end
%nolinear constraints
%(10c)
nonlcon = @rate_primal;
%----------------constraints end---------------------
Aeq = [];
Beq = [];
%solve problem
options = optimoptions('fmincon','MaxIterations',1e5,'MaxFunctionEvaluations',1e6);
[x,fval,exit_flag,~,lag] = fmincon(fun,x0,A,B,Aeq,Beq,lb,ub,nonlcon,options);
if exit_flag == -2
     p = zeros(K,L);
     eta = 0;
     lameda = zeros(L,1);
     miu = zeros(K,L);
     nu = zeros(K,L);
     theta = zeros(L,1);
else
    p = reshape(x(2:K*L+1),K,L);
    eta = -fval;
    [lameda,miu,nu,theta] = primal_dual_multiplier(K,L,P_max_D,rho,a,b,p_max,p,eta);
end
    

end