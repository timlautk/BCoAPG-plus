function i = co_update_l1(x,A,b,L,lambda,s,z)
[m,n] = size(A);
N = n/s;%block size

f = @(x) 1/(2*n)*norm(A*x-b,2)^2;
g = @(x) lambda*norm(x,1);

bvec = N*ones(s,1)';
C = mat2cell(A,m,bvec);
xj = mat2cell(x,bvec);
gradf = zeros(N,s);
for i = 1:s
    gradf(:,i) = 1/n*C{i}'*(A*x-b);
end

switch z
    case 0 %GS rule
        ngradf = zeros(s,1);
        for i = 1:s
            ngradf(i) = norm(gradf(:,i));
        end    
        [~,i] = max(abs(ngradf));
        
%     case 1 %GS-s rule
        
    case 2 %GS-r rule
        diff = zeros(N,s);
        for i = 1:s
            diff(:,i) = xj{i}-l1_prox(xj{i}-gradf(:,i)/L,lambda/L);
         end
        ndiff = zeros(s,1);
        for i = 1:s
            ndiff(i) = norm(diff(:,i));
        end
        [~,i] = max(ndiff);        
end
end