function i = co_update_scad(x,A,b,L,gamma,lambda,s,z)
[m,n] = size(A);
N = n/s;%block size

bvec = N*ones(s,1)';
C = mat2cell(A,m,bvec);
xj = mat2cell(x,bvec);
gradfj = zeros(N,s);
for i = 1:s
    gradfj(:,i) = 1/n*C{i}'*(A*x-b);
end

switch z
    case 0 %GS rule
        ngradf = zeros(s,1);
        for i = 1:s
            ngradf(i) = norm(gradfj(:,i));
        end    
        [~,i] = max(abs(ngradf));
        
%     case 1 %GS-s rule
        
    case 2 %GS-r rule
        diff = zeros(N,s);
        for i = 1:s
            diff(:,i) = xj{i}-scad_prox(xj{i}-gradfj(:,i)/L,gamma,lambda);
         end
        ndiff = zeros(s,1);
        for i = 1:s
            ndiff(i) = norm(diff(:,i));
        end
        [~,i] = max(ndiff);
end
end