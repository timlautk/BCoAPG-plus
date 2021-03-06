function val = bcapgncp_l1l2(A,b,lambda,s,beta,t,niter,fun,rule)

[~,n] = size(A);
Lf = 1/n*norm(A'*A,2);

x0 = zeros(n,1);
x = zeros(n,1);
v = zeros(n,1);
xhat = zeros(n,1);
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
            j = co_update_l2(x,A,b,Lf,lambda,s,2); %Gauss-Southwell Update, last arg z=0: GS; z=1: GS-s; z=2: GS-r
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
    x(J) = l2_prox(y, lambda/Lj);
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