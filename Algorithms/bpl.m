function val = bpl(varargin) %A,b,lambda,s,niter,fun,prox,theta

A = varargin{1};
b = varargin{2};
lambda = varargin{3};
s = varargin{4};
niter = varargin{5};
fun = varargin{6};
prox = varargin{7};
if nargin == 8 
    theta = varargin{8};
end 

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
    if nargin == 7
        x(J) = prox(y,lambda/Lj); % for Lasso
    elseif nargin == 8
        x(J) = prox(y,theta,lambda/Lj); % for Capped l1
    end
    t = (1+sqrt(1+4*t0^2))/2;
    omega = (t0-1)/t;
    t0 = t;
    val(k) = fun(x);
end
   fprintf('iteration:%d, objective value:%f\n',k,val(k));
end



end