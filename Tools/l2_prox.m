function p = l2_prox(x, lambda)
   p = max(0,1-lambda/norm(x,2)).*x; 
%     p = 1/(1+2*lambda)*x; 
end 