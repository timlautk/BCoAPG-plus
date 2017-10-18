function val = apgncp_scad(A,b,lambda,eta,beta,t,niter,gamma) 


[~,n] = size(A);
x = zeros(n,1);
xhat = zeros(n,1);

val = zeros(niter,1);
for k = 1:niter   
    gradf = 1/n*A'*(A*xhat-b);
    x0 = x;
    y = xhat - eta*gradf;
    x = scad_prox(xhat - eta*gradf,gamma,lambda);
    v = x + beta*(x-x0);
    val(k) = scad_fun(x,A,b,gamma,lambda);
    if scad_fun(x,A,b,gamma,lambda) <= scad_fun(v,A,b,gamma,lambda)
        xhat = x; beta = t*beta;
    else
        xhat = v; beta = min(beta/t,1);
    end
    fprintf('iteration:%d, objective value:%f\n',k,val(k));
end

end