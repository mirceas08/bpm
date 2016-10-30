function A = rmats(n, ls);

% generates a symmetric supersparse matrix
% with small integer entries
% n is matrix size, ls is size of local support
% ls should be <=10, a typical value would  be 4
% call: A = rmats( n, ls);

% generate ls indices 1 <= i1 < i2 <  < is <= n
p = randperm(n); p=p(1:ls); p=sort(p);
m = triu(rand(ls) -.5);
m = round(m*4);
m = m + m';
%m = rmat(ls,-3);   % coefficients in range [-3, ..., +3]
A=zeros(n);
A(p,p)=m; 
A=sparse(A);
