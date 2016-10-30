size = 100;
%blocks = 6;

nods = size-1;
density = 0.5;

A = graph(nods, density);

% find_clique function can be used
clique = [1:19];
len_clique = length(clique);

A(clique,clique) = ones(len_clique) - eye(len_clique);

tic
data = psi_bound_data(A, clique);
toc
