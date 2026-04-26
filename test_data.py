import numpy as np

ver=0
A_data = np.load(f'data/A_data_{ver}.npy', allow_pickle=True)
# print(A_data)
print(A_data.shape)

B_data = np.load(f'data/A_data_{ver}.npy', allow_pickle=True)
print(B_data)
print(B_data.shape)