function i = co_update_capped_l1(x,A,b,L,lambda,theta,s,z)
[m,n] = size(A);
N = n/s;%block size

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
            diff(:,i) = xj{i}-cap_l1_prox(xj{i}-gradf(:,i)/L,theta,lambda/L);
         end
        ndiff = zeros(s,1);
        for i = 1:s
            ndiff(i) = norm(diff(:,i));
        end
        [~,i] = max(ndiff);
end
end