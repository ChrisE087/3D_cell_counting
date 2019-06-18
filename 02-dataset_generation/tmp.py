import nrrd
import matplotlib.pyplot as plt

data, header = nrrd.read('dataset\24h_C2-untreated_1.1-1640.nnrd')
X = data[0,]
y = data[1,]

plt.imshow(X[:,:,25])
plt.imshow(y[:,:,25])
