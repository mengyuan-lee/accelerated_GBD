function [c,ceq] = rate_primal(x)

load const.mat
R = zeros(L,1);


for i = 1:L
    for j = 1:K
        R(i) =  R(i) + log(1+x(1+(i-1)*K+j)/(a(j,i)+ b(j,i)*x(1+(i-1)*K+j)))/log(2);
    end
end
c = ones(L,1)*x(1)-R;
ceq = [];
end