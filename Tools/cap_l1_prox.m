function p = cap_l1_prox(u,theta,lambda)
    p = zeros(size(u));
    p1 = sign(u).*max(theta,abs(u));
    p2 = sign(u).*min(theta,max(0,abs(u)-lambda));
    h = @(x) 0.5*(x-u).^2+lambda*min(abs(x),theta);

    p(h(p1) <= h(p2)) = p1(h(p1) <= h(p2));
    p(h(p1) > h(p2)) = p2(h(p1) > h(p2));
end