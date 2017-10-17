function val = apg(varargin) %A,b,lambda,eta,niter,fun,prox,theta

A = varargin{1};
b = varargin{2};
lambda = varargin{3};
eta = varargin{4};
niter = varargin{5};
fun = varargin{6};
prox = varargin{7};
if nargin == 8 
    theta = varargin{8};
end 

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
    if nargin == 7
        x = prox(y,lambda*eta); % for Lasso
    elseif nargin == 8
        x = prox(y,theta,lambda*eta); % for Capped l1
    end
    t = (1+sqrt(1+4*t0^2))/2;
    omega = (t0-1)/t;
    t0 = t;
    val(k) = fun(x);
    fprintf('iteration:%d, objective value:%f\n',k,val(k));
end

end