function fb = f_banco_filtros_mel(F,B, fs)
% fb = f_banco_filtros_mel(F,B, fs)
% Calcula la matriz de un banco de filtros en escala Mel.
%
% F: NFFT / 2
% B: Número de filtros
% fs: frecuencia de muestreo

	StFreq = 64.0;
	fb = zeros(F,B);
    %/* Constants for calculation*/
    start_mel    = 2595.0 * log10 (1.0 + StFreq / 700.0);   
    fs_per_2_mel = 2595.0 * log10 (1.0 + (fs / 2) / 700.0);
    for b = 0:B-1
        %/* Calculating mel-scaled frequency and the corresponding FFT-bin */
        %/* number for the lower edge of the band                          */
        freq = 700 * (10^ ( (start_mel + b / (B + 1) * (fs_per_2_mel - start_mel)) / 2595.0) - 1.0);
        f1 = (2 * F * freq / fs + 0.5);

        %/* Calculating mel-scaled frequency for the upper edge of the band */
        freq = 700 * (10^ ( (start_mel + (b + 2) / (B + 1) * (fs_per_2_mel - start_mel)) / 2595.0) - 1.0);

        %/* Calculating and storing the length of the band in terms of FFT-bins*/
		f3 = (2 * F * freq / fs + 0.5);
		f2 = (f1 + f3)/2;
		f3 = min(f3,F-1);

 		s =0.0;
		f1 = floor(f1);
		f2 = floor(f2);
		f3 = floor(f3);		
		for f=f1:f2-1
	  		fb(f+1,b+1) = f*(1/(f2-f1)) - f1/(f2-f1);
  	  		s = s + fb(f+1,b+1);
		end		
		for f=f2:f3-1
			fb(f+1,b+1) = f*(-1/(f3-f2)) + f3/(f3-f2);
   	  		s = s + fb(f+1,b+1);
		end
    end
end