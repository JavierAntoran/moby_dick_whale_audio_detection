import numpy as np
import matplotlib.pyplot as plt

details = np.load('data/processed_data_swt_dct32_details.npy')
approx = np.load('data/processed_data_swt_dct32.npy')

Nvis = 0

ap = approx[Nvis]
de = details[Nvis]

plt.figure(dpi=80)
plt.imshow(ap.T, cmap='jet', aspect='auto')
plt.title('swt approximation multiresolution dct')
plt.gca().invert_yaxis()
plt.ylabel('coefficients')
plt.xlabel('N window')
plt.show()
plt.figure(     dpi=80)
plt.imshow(de.T, cmap='jet', aspect='auto')
plt.title('swt details multiresolution dct')
plt.gca().invert_yaxis()
plt.ylabel('coefficients')
plt.xlabel('N window')
plt.show()