function b = f_base_dct(N)
% b = f_base_dct(N)
% Calcula la matriz de la base Discrete Cosine Transform con los vectores
% almacenados por columnas.
%
% N: dimension de la base

    b = zeros(N,N);
    for n=0:N-1
       if (n==0)
          kn = sqrt(1/N);
       else
          kn = sqrt(2/N);
       end
       for m=0:N-1
          b(m+1,n+1) = kn * cos( (2*m + 1)*n*pi / (2*N) );
       end
    end
end