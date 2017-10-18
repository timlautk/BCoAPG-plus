function val = bpl_l1l2(A,b,lambda,s,niter,fun) 

[~,n] = size(A);
x0 = zeros(n,1);
x = zeros(n,1);
omega = 0;
t0 = 1;
val = zeros(niter,1);
N = n/s;

for k = 1:niter
for i = 1:s
    j = unidrnd(s); % uniformly random update
    J = (j-1)*N+1:j*N;
    xhat = x(J) + omega*(x(J) - x0(J));
    Abar = A;
    Abar(:,J) = [];
    xbar = x;
    xbar(J) = [];
    Aj = A(:,J);
    Lj = 1/n*norm(Aj'*Aj);
    x0(J) = x(J);
    gradf = 1/n*Aj'*(Aj*xhat+Abar*xbar-b);
    y = xhat-gradf/Lj;
    x(J) = l2_prox(y, lambda/Lj);
    t = (1+sqrt(1+4*t0^2))/2;
    omega = (t0-1)/t;
    t0 = t;
    if fun(x) > fun(x0)
        omega = omega/2;
    end
    if omega < 1e-2
        omega = 0;
    end
    val(k) = fun(x);
end
    fprintf('iteration:%d, objective value:%f\n',k,val(k));
end

end