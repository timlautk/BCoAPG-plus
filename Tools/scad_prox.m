function p = scad_prox(u,gamma,lambda)
    p = zeros(size(u));
    
    p1 = sign(u).*min(lambda,max(0,abs(u)-lambda));
    p2 = sign(u).*min(gamma*lambda,max(lambda,(abs(u)*(gamma-1)-gamma*lambda)/(gamma-2)));
    p3 = sign(u).*max(gamma*lambda,abs(u));
    
    
    p((h(p1) < h(p2)) & (h(p1) < h(p3))) = p1((h(p1) < h(p2)) & (h(p1) < h(p3)));
    p((h(p2) < h(p1)) & (h(p2) < h(p3))) = p2((h(p2) < h(p1)) & (h(p2) < h(p3)));
    p((h(p3) < h(p1)) & (h(p3) < h(p2))) = p3((h(p3) < h(p1)) & (h(p3) < h(p2)));

function val = h(x)
    val = zeros(size(x));
    h1 = 0.5*(x-u).^2+lambda*abs(x);
    h2 = 0.5*(x-u).^2+(2*gamma*lambda*abs(x)-x.^2-lambda^2)/(gamma-1);
    h3 = 0.5*(x-u).^2+lambda^2*(gamma^2-1)/(2*(gamma-1));
    
    val(abs(x) <= lambda) = h1(abs(x) <= lambda);
    val(abs(x) > lambda & abs(x) <= gamma*lambda) = h2(abs(x) > lambda & abs(x) <= gamma*lambda);
    val(abs(x) > gamma*lambda) = h3(abs(x) > gamma*lambda);
end
end