function [data] = psi_bound_data(A,K);
% Generates data for Psi_k(G) bound for \Chi(G)
% defined as: 
%  Psi_K(G) = \min X00
%  s.t.   X^0_i0=X^0_ii=1
%         X^i_{j,k}=0 if {i,j,k} contains at least one edge, where
%         {i}\subset K, i,j \in V
%          X ... PSD
% Equivalent formulation
%  Psi_K(G) = \min X00
%  s.t.   X^0_i0=X^0_ii=1
%         X^i_{j,k}=0 if {i,j,k} contains at least one edge, where
%         {i}\subset K, i,j \in V
%
% Important: for each {i,j} \subset K, i\neq j
%             X^i_{jj}=0, hence we delete j-th row and column
% K \subset V(G) defining a clique in graph G
% A ... adj. matrix of graph G
% solver...name od the solver tu use
% data  ... data describing block diagonal SDP for bpm_block
% 
% created:      2015-7-09 by JP
%% CALL:   [data] = psi_bound_new(A,K,solver);

L_G=[]; % list of edges
for i=1:size(A,1) 
  for j=i+1:size(A,1) 
    if A(i,j) 
      L_G=[L_G; i j]; 
    end;
  end;
end;
n=size(A,1);
m=size(L_G,1);

k=length(K);
tot_num = 3*k+m*k+n+m+k;

A_G=A;

blk=cell(k+1,2);
for i=1:k+1
  blk{i,1}='s';
  blk{i,2}=n+1;
end;
A=cell(k+1,1);
b=[];

C=cell(k+1,1);
C{1}=sparse(1,1,1,n+1,n+1);
for i=2:k+1
  C{i}=sparse(n+1,n+1);
end;

count = 1;
for i=2:k+1   % X^i_0,i = X^i_i,i = X^i_0,0 = 1;  i\in K  - 3*k contraints
  A1=sparse(1,1,1,n+1,n+1); % X^i_00 = 1
  A2 = sparse([1 K(i-1)+1],[K(i-1)+1 1],[0.5 0.5],n+1,n+1);  % X^i_{0i}=1
  A3 = sparse(K(i-1)+1,K(i-1)+1,1,n+1,n+1); % X^i_ii = 1
  for j=1:k+1
    if i==j
      A{j,count}=A1;
      A{j,count+1}=A2;
      A{j,count+2}=A3;    
    else
      A{j,count}=sparse(n+1,n+1);
      A{j,count+1}=sparse(n+1,n+1);
      A{j,count+2}=sparse(n+1,n+1);    
    end; 
    b=[b;ones(3,1)];
  end;
  count=count+3;
end;
count = count-1;

for i=1:k+1  % X^{i}_{jk} = 0 if {i,j,k} contains an edge 
           % e stays for edges
  for j=1:n+1
    for el = j:n+1
      A1=[];A2=[];
      if (i==j && i==el) || (j+el==2) || (i+j==2) % there is no edge in {i,j,el}
        continue;
      elseif i==1 && A_G(j-1,el-1)  
        % (j,el) is an edge, hence X^0_{j,el}=0
        A1=sparse([j el],[el j],[1 1],n+1,n+1);        
      elseif i > 1 && j==1 && A_G(K(i-1),el-1) 
        % (i,el) is an edge, hence X^i{0,el}=X^i{el,el}=0
        A1=sparse([1 el],[el 1],[1 1],n+1,n+1);
        A2=sparse(el,el,1,n+1,n+1);
      elseif i > 1  && j > 1 && A_G(j-1,el-1)
        % (j,el) is an edge, 
        %  hence X^i{j,el}=0
        A1=sparse([j el],[el j],[1 1],n+1,n+1);        
      end;  
      if ~isempty(A1)
        count=count+1;
        A{i,count}=A1;
        b(count,1)=0;
        for p=1:k+1  % we add Ai for other blocks
          if p~=i
            A{p,count}=sparse(n+1,n+1);
          end;
        end;
      end;
      if ~isempty(A2)
        count=count+1;
        A{i,count}=A2;
        b(count,1)=0;
        for p=1:k+1  % we add Ai for other blocks
          if p~=i
            A{p,count}=sparse(n+1,n+1);
          end;
        end;
      end;  % If
    end; % for
  end; %for
end; %for

% first block

for j=2:n+1  % X^0_{jj}+\sum_s X^s_{jj}=1
  Aj=sparse(j,j,1,n+1,n+1);
  count=count+1;
  for p=1:k+1  % we define the equation
    A{p,count}=Aj;
  end;
  b(count,1)=1;
end;

%  X^i_{j,j}=X^i_{i,j}=X^i_{0,j}, if (i,j) not edge,i,j>1,i\neq j
for i=2:k+1
  for j=2:n+1
    A1=[];A2=[];A3=[];
    if K(i-1)~=j-1 && ~A_G(K(i-1),j-1)  % (i,j) is not edge
      A1=sparse([j K(i-1)+1],[j j],[1 -1],n+1,n+1);
      A1=A1+A1';  % X^i_j,j=X^i_i,j
      A2=sparse([j 1],[j j],[1 -1],n+1,n+1);
      A2=A2+A2'; % X^i_0,j=X^i_j,j
      for el=1:k+1
        b(count+1,1)=0;        
        b(count+2,1)=0;
        if el~=i
          A{el,count+1}=sparse(n+1,n+1);
          A{el,count+2}=sparse(n+1,n+1);
        else
          A{el,count+1}=A1;
          A{el,count+2}=A2;
        end;          
      end;
      count=count+2;
    end;  
  end;
  
end;
% X^0_{i,i}=X^0_{0,i}, i not in a clique K
for i=2:n+1
  if isempty(K) || ~max(K==i-1)  % i is not in a clique
    count = count +1;
    A{1,count}=sparse([1 i i],[i 1 i],[-1 -1 2],n+1,n+1);
    b(count,1)=0;
    for p=2:k+1
      A{p,count}=sparse(n+1,n+1);
    end;  
  end;
end;

data.dim=(n+1)*ones(k+1,1);
% a  vector with dimensions  of blocks on the diagonal 
data.A_GT=cell(k+1,1);
data.C=cell(k+1,1);
data.AAT=sparse(count,count);
for i = 1:k+1 
  A_temp=sparse((n+1)^2,size(b,1));
  data.C{i}=-C{i}; 
  for j=1:size(b,1)
    A_temp(:,j)=A{i,j}(:);  %A_temp is essentially A^T for i-th block
  end; 
  data.A_GT{i}=A_temp;
  data.AAT=data.AAT+A_temp'*data.A_GT{i};
end;
pars.sigma       = 1;  % default sigma
pars.tol         = 1e-5; % default tolerance
pars.max_iter    = 600;  % default max_iter
pars.in_max_init = 1;    % default number of inner iterations at the  
pars.in_max_inc  = 0;    % number of inner iterations increased by this
                       % value in each outer iteration
pars.in_max_max  = 1;    % maximum number of inner iterations
pars.print_iter  = 10;   % print every 10th iteration

data.b=b;

%%%%%%%%% Saving data to file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

path = './';
delete(strcat(path, 'data/*.dat'));

for i = 1:k+1
  A_GT_temp = data.A_GT{i};
  filenameA = strcat(path, 'data/A_GT', num2str(i), '.dat');
  writesparse(A_GT_temp, filenameA);
  C_temp = data.C{i};  
  filenameC = strcat(path, 'data/C', num2str(i), '.dat');
  writesparse(C_temp, filenameC);
end

nonzero = nnz(data.AAT);
for i = 1:k+1
  nonzero = [nonzero; nnz(data.A_GT{i})];  
end

for i = 1:k+1
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


