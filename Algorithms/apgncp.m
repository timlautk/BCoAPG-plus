function val = apgncp(A,b,lambda,eta,beta,t,niter,fun,prox,varargin) 

if nargin == 10 
    theta = varargin{1};
end 

[~,n] = size(A);
x = zeros(n,1);
xhat = zeros(n,1);

val = zeros(niter,1);
for k = 1:niter   
    gradf = 1/n*A'*(A*xhat-b);
    x0 = x;
    y = xhat - eta*gradf;
    if nargin == 9
        x = prox(y,lambda*eta); % for Lasso
    elseif nargin == 10
        x = prox(y,theta,lambda*eta); % for Capped l1
    end
    v = x + beta*(x-x0);
    val(k) = fun(x);
    if fun(x) <= fun(v) 
        xhat = x; beta = t*beta;
    else
        xhat = v; beta = min(beta/t,1);
    end
    fprintf('iteration:%d, objective value:%f\n',k,val(k));
end

end