function val = bcapgncp(varargin) %A,b,lambda,s,beta,t,niter,fun,prox,rule,co_update,theta

A = varargin{1};
b = varargin{2};
lambda = varargin{3};
s = varargin{4};
beta = varargin{5};
t = varargin{6};
niter = varargin{7};
fun = varargin{8};
prox = varargin{9};
rule = varargin{10};
co_update = varargin{11};
if nargin == 12 
    theta = varargin{12};
end 

[~,n] = size(A);
Lf = 1/n*norm(A'*A,2);

x0 = zeros(n,1);
x = zeros(n,1);
xhat = zeros(n,1);
v = zeros(n,1);
val = zeros(niter,1);
t = t*ones(s,1); 
beta = beta*ones(s,1); 
N = n/s; %block size

for k = 1:niter
for i = 1:s
    switch rule 
        case 1
            j = unidrnd(s); % uniformly random update
        case 2
            if nargin > 11
                j = co_update(x,A,b,Lf,lambda,theta,s,2);
            else
                j = co_update(x,A,b,Lf,lambda,s,2); %Gauss-Southwell Update, last arg z=0: GS; z=1: GS-s
            end
    end
    J = (j-1)*N+1:j*N;
    xhat(J) = x(J) + beta(j)*(x(J) - x0(J));
    Abar = A;
    Abar(:,J) = [];
    xbar = x;
    xbar(J) = [];
    Aj = A(:,J);
    Lj = 1/n*norm(Aj'*Aj);
    gradf = 1/n*Aj'*(Aj*xhat(J)+Abar*xbar-b);
    x0(J) = x(J);
    y = xhat(J) - gradf/Lj;
    if nargin > 11
        x(J) = prox(y,theta,lambda/Lj); % for Capped l1
    else
        x(J) = prox(y,lambda/Lj); % for Lasso
    end
    v(J) = x(J) + beta(j)*(x(J)-x0(J));
    val(k) = fun(x);
    if fun(x) <= fun(v) 
        beta(j) = t(j)*beta(j);
    else

        beta(j) = min(beta(j)/t(j),1);
    end
end
   fprintf('iteration:%d, objective value:%f\n',k,val(k));
end

end