function val = apgnc(A,b,lambda,eta,niter,fun,prox,varargin)

if nargin == 8 
    theta = varargin{1};
end 

[~,n] = size(A);
x = zeros(n,1);
xhat = zeros(n,1);
val = zeros(niter,1);
for k = 1:niter   
    beta = k/(k+3);
    gradf = 1/n*A'*(A*xhat-b);
    x0 = x;
    y = xhat - eta*gradf;
    if nargin == 7
        x = prox(y,lambda*eta); % for Lasso
    elseif nargin == 8
        x = prox(y,theta,lambda*eta); % for Capped l1
    end
    v = x + beta*(x-x0);
    val(k) = fun(x);
    if fun(x) <= fun(v)
        xhat = x;
    else
        xhat = v;
    end
    fprintf('iteration:%d, objective value:%f\n',k,val(k));
end

end