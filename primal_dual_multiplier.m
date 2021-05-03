function [lameda,miu,nu,theta] = primal_dual_multiplier(K,L,P_max_D,rho,a,b,p_max,p,eta)

A = zeros(2*L+3*K*L+1,2*L+2*K*L);
B = zeros(2*L+3*K*L+1,1);
lb = zeros(2*L+2*K*L,1);

temp1 = sum(p) - P_max_D;
for i=1:L
    A(i,i)=temp1(i);
end

temp2 = p - rho.*p_max;
for i=1:L
    for j=1:K
        A(L+(i-1)*K+j,L+(i-1)*K+j)=temp2(j,i);
    end
end

temp3 = -p;
for i=1:L
    for j=1:K
        A(L+K*L+(i-1)*K+j,L+K*L+(i-1)*K+j)=temp3(j,i);
    end
end

temp4 = eta - sum_log(1+(1-a.*inv_pos(a+b.*p))./b)/log(2);
for i=1:L
    A(L+2*K*L+i,L+2*K*L+i)=temp4(i);
end

for i=1:L
    for j=1:K
        A(2*L+2*K*L+(i-1)*K+j,i)=1;
        A(2*L+2*K*L+(i-1)*K+j,L+(i-1)*K+j)=1;
        A(2*L+2*K*L+(i-1)*K+j,L+K*L+(i-1)*K+j)=-1;
        A(2*L+2*K*L+(i-1)*K+j,L+2*K*L+i)=-a(j,i)/log(2)/((a(j,i)+b(j,i)*p(j,i))^2+p(j,i)*(a(j,i)+b(j,i)*p(j,i)));
    end
end

A(2*L+3*K*L+1,L+2*K*L+1:L+2*K*L+L)=1;
B(2*L+3*K*L+1,1)=1;
    

x = lsqlin(A,B,[],[],[],[],lb);
lameda = x(1:L);
miu = reshape(x(L+1:L+K*L),K,L);
nu = reshape(x(L+K*L+1:L+2*K*L),K,L);
theta = x(L+2*K*L+1:2*L+2*K*L);


end