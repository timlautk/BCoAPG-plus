function val = apgnc_scad(A,b,lambda,eta,niter,gamma)

[~,n] = size(A);
x = zeros(n,1);
xhat = zeros(n,1);
val = zeros(niter,1);
for k = 1:niter   
    beta = k/(k+3);
    gradf = 1/n*A'*(A*xhat-b);
    x0 = x;
    x = scad_prox(xhat - eta*gradf,gamma,lambda);
    v = x + beta*(x-x0); 
    if scad_fun(x,A,b,gamma,lambda) <= scad_fun(v,A,b,gamma,lambda)
        xhat = x;
    else
        xhat = v;
    end
    val(k) = scad_fun(x,A,b,gamma,lambda);
    fprintf('iteration:%d, objective value:%f\n',k,val(k));
end

end