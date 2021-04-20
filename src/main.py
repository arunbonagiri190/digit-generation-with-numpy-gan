from GAN import Generator, Discriminator
import util
from util import DataLoader

# load data
print('main | Initializing ... ')
digits, noise = DataLoader.load_data()
D = Discriminator()
G = Generator()

# train GAN
print('main | Training ... ')
epochs = 7000
dErrors = []
gErrors = []

for epoch in range(epochs):
    d_error1 , d_error2 , g_error = 0, 0, 0
    for digit in digits:
        d_error1 += D.fit(digit, isDigit=True)
        gOut = G.generate()
        d_error2 = D.fit(gOut)
        g_error += G.fit(gOut, D)

    if (epoch % 100) == 0:
        dErrors.append(((d_error1 + d_error2)/2)/14)
        gErrors.append(g_error/14)

# show results
sprt = [i for i in range(epochs//100)] # for x-axis
util.save_png(G.generate(), "gen_image_ephocs_"+str(epochs))
util.save_plot(gErrors, dErrors, sprt, "error_plot_ephocs_"+str(epochs))
print('main | Completed.')