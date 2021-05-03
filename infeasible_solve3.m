function [p,eta,alphaMy,lameda,miu,nu,theta] = infeasible_solve3(K,L,R_min_C,P_max_D,P_max_C,h_CD,h_D,h_CB,h_DB,rho)
%output: exit_flag: Reason fmincon stopped, returned as an integer.
%        p eta alpha1 alpha2 alpha3
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

[a,b,p_max] = para(K,L,R_min_C,P_max_D,P_max_C,h_CD,h_D,h_CB,h_DB);
cvx_begin
    variables alphaMy eta p(K,L)
    dual variables lameda miu nu theta
    minimize alphaMy
    subject to
        lameda: sum(p) <= P_max_D + alphaMy
        miu: p <= rho.*p_max + alphaMy
        nu: 0 <= p + alphaMy
        theta: eta <= sum_log(1+(1-inv_pos(1+b./a.*p))./b)/log(2)+alphaMy
        %theta: eta <= sum(1000/log(2)*(((1-a.*inv_pos(a+b.*p))./b).^(1/1000)-1))+alphaMy
        0 <= alphaMy
        0<= eta
cvx_end




p = full(p);
lameda = reshape(full(lameda),L,1);
miu =  full(miu);
nu = full(nu);
theta = reshape(full(theta),L,1);

end