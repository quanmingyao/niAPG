clear; clc;

maxNumCompThreads(1);

load('data/movielens1m.mat');

[row, col, val] = find(data);

[m, n] = size(data);

clear user item;

val = val - mean(val);
val = val/std(val);

idx = randperm(length(val));

traIdx = idx(1:floor(length(val)*0.5));
tstIdx = idx(ceil(length(val)*0.5): end);

clear idx;

traData = sparse(row(traIdx), col(traIdx), val(traIdx));
traData(size(data,1), size(data,2)) = 0;

para.test.row  = row(tstIdx);
para.test.col  = col(tstIdx);
para.test.data = val(tstIdx);
para.test.m = m;
para.test.n = n;

clear m n;

%%
lambda = 2000;
theta = sqrt(lambda);
  
para.maxR = 5;
para.regType = 2;
para.maxIter = 1000;
para.tol = 1e-5;

% APGnc(exact)
method = 1;
[~, ~, ~, out{method}] = APGncext( traData, lambda, theta, para );

% APGnc
method = 2;
[~, ~, ~, out{method}] = APGnc( traData, lambda, theta, para );

close all;
for i = 1:method
    plot(out{i}.Time, out{i}.RMSE);
    hold on;
end
legend('niAPG(exact)', 'niAPG');







