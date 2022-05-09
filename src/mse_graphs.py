import matplotlib.pyplot as plt
import matplotlib
import numpy as np 
from skimage.filters import gaussian, unsharp_mask
import scipy
matplotlib.rcParams.update({'font.size': 15})

def highpass(filepath, sigma):
    # High Pass
    img = np.load(filepath)
    low_pass = gaussian(img, sigma=sigma)
    high_pass = np.subtract(img,(0.8*low_pass))
    mean, std = (np.mean(high_pass), np.std(high_pass))

    return high_pass, mean, std

def monoExp(x, m, t, b):
    return (m * np.exp((-t) * x)) + b

# Composite Image
highpass("../log/composite_image_1.npy", 20)
compositeimage1, mean1, std1 = highpass("../log/composite_image_1.npy", 20)
compositeimage3, mean3, std3 = highpass("../log/composite_image_3.npy", 20)
compositeimage5, mean5, std5 = highpass("../log/composite_image_5.npy", 20)
compositeimage7, mean7, std7 = highpass("../log/composite_image_7.npy", 20)
compositeimage9, mean9, std9 = highpass("../log/composite_image_9.npy", 20)
compositeimage11, mean11, std11 = highpass("../log/composite_image_11.npy", 20)

mse1 = np.square(np.subtract(compositeimage1, compositeimage11)).mean()
mse3 = np.square(np.subtract(compositeimage3, compositeimage11)).mean()
mse5 = np.square(np.subtract(compositeimage5, compositeimage11)).mean()
mse7 = np.square(np.subtract(compositeimage7, compositeimage11)).mean()
mse9 = np.square(np.subtract(compositeimage9, compositeimage11)).mean()
mse11 = np.square(np.subtract(compositeimage11, compositeimage11)).mean()

x = np.asarray([1, 8, 24, 48, 80, 120])
y = np.asarray([mse1, mse3, mse5, mse7, mse9, mse11])


p0 = (2000, .1, 50) # start with values near those we expect
params, cv = scipy.optimize.curve_fit(monoExp, x, y, p0)
m, t, b = params

# determine quality of the fit
squaredDiffs = np.square(y - monoExp(x, m, t, b))
squaredDiffsFromMean = np.square(y - np.mean(y))
rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
print(f"R² = {rSquared}")

x_many = np.linspace(1, 120, 120)



plt.figure(2)
plt.xlabel(r"Number of Images used in Reconstruction")
plt.ylabel(r"Image MSE")
plt.scatter(x,y, label='actual', marker='^', linewidth=5)
plt.plot(x_many, monoExp(x_many, m, t, b), '--', label="fitted")
plt.annotate("r-squared = {:.3f}".format(rSquared), xy=(75, 0.5))
plt.annotate("Upsampled 1x1", xy=(6, 0.675))
plt.annotate("3x3", xy=(12, 0.305))
plt.annotate("5x5", xy=(23, 0.06))
plt.annotate("7x7", xy=(45, 0.035))
plt.annotate("9x9", xy=(75, 0.03))
plt.annotate("11x11", xy=(110, 0.025))

plt.legend()
plt.show()

# fig, ax = plt.subplots(1,6)
# ax[0].imshow(compositeimage1, cmap='Greys', vmin=mean1-(8*std1), vmax=mean1+(8*std1))
# ax[1].imshow(compositeimage3, cmap='Greys', vmin=mean3-(8*std3), vmax=mean3+(8*std3))
# ax[2].imshow(compositeimage5, cmap='Greys', vmin=mean5-(8*std5), vmax=mean5+(8*std5))
# ax[3].imshow(compositeimage7, cmap='Greys', vmin=mean7-(8*std7), vmax=mean7+(8*std7))
# ax[4].imshow(compositeimage9, cmap='Greys', vmin=mean9-(8*std9), vmax=mean9+(8*std9))
# ax[5].imshow(compositeimage11, cmap='Greys', vmin=mean11-(8*std11), vmax=mean11+(8*std11))

# ax[0].set_xlabel("1x1")
# ax[1].set_xlabel("3x3")
# ax[2].set_xlabel("5x5")
# ax[3].set_xlabel("7x7")
# ax[4].set_xlabel("9x9")
# ax[5].set_xlabel("11x11")
# plt.show()

