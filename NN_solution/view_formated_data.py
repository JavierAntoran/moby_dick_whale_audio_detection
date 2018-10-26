import numpy as np
import matplotlib.pyplot as plt

details = np.load('data/processed_data_swt_details.npy')
approx = np.load('data/processed_data_swt_approx.npy')

Nvis = 7

ap = approx[Nvis]
de = details[Nvis]

plt.figure(dpi=80)
plt.imshow(ap.T, cmap='jet', aspect='auto')
plt.title('swt approximation multiresolution dct')
plt.gca().invert_yaxis()
plt.ylabel('coefficients')
plt.xlabel('N window')
# plt.savefig('swt_approximation_dct.png')
plt.show()
plt.figure(dpi=80)
plt.imshow(de.T, cmap='jet', aspect='auto')
plt.title('swt details multiresolution dct')
plt.gca().invert_yaxis()
plt.ylabel('coefficients')
plt.xlabel('N window')
# plt.savefig('swt_details_dct.png')
plt.show()