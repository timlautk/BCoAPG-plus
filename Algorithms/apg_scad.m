function val = apg_scad(A,b,lambda,eta,niter,gamma)


[~,n] = size(A);
x0 = zeros(n,1);
x = zeros(n,1);
t0 = 1;
omega = 0;
val = zeros(niter,1);

for k = 1:niter   
    xhat = x + omega*(x - x0);
    gradf = 1/n*A'*(A*xhat-b);
    x0 = x;
    y = xhat - eta*gradf;
    x = scad_prox(y,gamma,lambda);
    t = (1+sqrt(1+4*t0^2))/2;
    omega = (t0-1)/t;
    t0 = t;
    val(k) = scad_fun(x,A,b,gamma,lambda);
    fprintf('iteration:%d, objective value:%f\n',k,val(k));
end

end