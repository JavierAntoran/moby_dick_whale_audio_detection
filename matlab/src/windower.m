function [X] = windower(x, M, N)
% M avance entre vetanas
% N windowsize

T   = length(x);
m   = 1:M:T-N; % comienzos de ventana
L   = length(m);% N ventanas
ind = (0:N-1)' * ones(1,L) + ones(N,1) * m;
X   = x(ind);

end

