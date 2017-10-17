function [A,b,eta] = data_gen_scad(m,n)

rng('default');
seed = 20;
rng(seed);

A = randn(m,n);
A = A-mean(A);
A = A./std(A);
b = randn(m,1);
b = b-mean(b);

Lf = 1/n*norm(A'*A);
eta = 1/Lf;

end