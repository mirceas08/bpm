%%%%%%%%%%% miscellaneous
data.b = SDP_data.b;
data.dim = SDP_data.K.s;
p = length(data.dim);
A_sparse = sparse(SDP_data.A);
data.AAT = A_sparse' * A_sparse;    

%%%%%%%%%% data.A_GT
data.A_GT{1} = A_sparse(1:sum(data.dim(1:1).^2),:);
for i=2:p
    data.A_GT{i} = A_sparse(sum(data.dim(1:i-1).^2) + 1:sum(data.dim(1:i).^2),:);
end

%%%%%%%%%% data.C
data.C{1} = SDP_data.C(1:sum(data.dim(1:1))^2);
data.C{1} = reshape(data.C{1}, data.dim(1), data.dim(1));
for i=2:p
    data.C{i} = SDP_data.C(sum(data.dim(1:i-1).^2) + 1:sum(data.dim(1:i).^2));
    data.C{i} = reshape(data.C{i}, data.dim(i), data.dim(i));
end

%%%%%%%%% Saving data to file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

path = './';
delete(strcat(path, 'data/*.dat'));

for i = 1:p
  A_GT_temp = data.A_GT{i};
  filenameA = strcat(path, 'data/A_GT', num2str(i), '.dat');
  writesparse(A_GT_temp, filenameA);
  C_temp = data.C{i};  
  filenameC = strcat(path, 'data/C', num2str(i), '.dat');
  writesparse(C_temp, filenameC);
end

nonzero = nnz(data.AAT);
for i = 1:p
  nonzero = [nonzero; nnz(data.A_GT{i})];  
end

for i = 1:p
  nonzero = [nonzero; nnz(data.C{i})];  
end
 
b_temp = data.b;
AAT_temp = data.AAT;
dim_temp = data.dim;

size(data.b)
writesparse(AAT_temp, strcat(path, 'data/AAT.dat'));
save(strcat(path, 'data/b.dat'), 'b_temp', '-ascii');
save(strcat(path, 'data/dim.dat'), 'dim_temp', '-ascii');
save(strcat(path, 'data/nonzeros.dat'), 'nonzero', '-ascii');
