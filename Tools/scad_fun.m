function val = scad_fun(beta,X,y,gamma,lambda)
    [n,p] = size(X);
    r = zeros(p,1);
    x1 = lambda*abs(beta);
    x2 = (2*gamma*lambda*abs(beta)-beta.^2-lambda^2)/(gamma-1);
    x3 = lambda^2*(gamma^2-1)/(2*(gamma-1))*ones(p,1);
    
    r(abs(beta) <= lambda) = x1(abs(beta) <= lambda);
    r((abs(beta) > lambda) & (abs(beta) <= gamma*lambda)) = x2((abs(beta) > lambda) & (abs(beta) <= gamma*lambda));
    r(abs(beta) > gamma*lambda) = x3(abs(beta) > gamma*lambda);
    
    val = 1/(2*n)*norm(X*beta-y)^2+sum(r);
end