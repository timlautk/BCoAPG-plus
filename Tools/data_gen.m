function [A,b,eta] = data_gen(m,n)

rng('default');
seed = 20;
rng(seed);

A = randn(m,n);
A = A+1;
A = 10*randn(m,1).*A;
S = sprandn(m,n,10*log(n)/n);
A(S == 0) = 0;
v0 = randn(n,1);
e = sqrt(0.1)*randn(m,1);
b = A*v0+e;

Lf = 1/n*norm(A'*A,2);
eta = 1/Lf;

end