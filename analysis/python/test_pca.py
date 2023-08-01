import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import scipy
import scipy.linalg


arr = numpy.random.multivariate_normal(
    mean = (300, 50),
    cov = [[55, 50], [50, 100]],
    size = 2000
)

x = arr[:, 0]
y = arr[:, 1]
c = numpy.random.random(size = len(x))

print()

fig = plt.figure(figsize = [8, 5])
colormap = mpl.cm.get_cmap("nipy_spectral").copy()
ax = fig.add_subplot(1, 1, 1)

mean_x = numpy.average(x)
mean_y = numpy.average(y)

cov_xx = numpy.average((x - mean_x)**2)
cov_yy = numpy.average((y - mean_y)**2)
cov_xy = numpy.average((x - mean_x)*(y-mean_y))

covmat = numpy.array([
    [cov_xx, cov_xy],
    [cov_xy, cov_yy],
])

eigvals, eigvecs = scipy.linalg.eig(covmat)#, left = True, right = True)
eig1 = eigvecs[0]
eig2 = eigvecs[1]

im = ax.scatter(
    x = x,#-mean_x,
    y = y,#-mean_y,
    c = c,
    cmap = colormap,
)

fig.colorbar(
    mappable = im,
    ax = ax,
    label = "weight",
    location = "right",
    orientation="vertical",
)

line_x = [min(x), max(x)]

# Convert vector to line
# https://math.libretexts.org/Bookshelves/Calculus/CLP-3_Multivariable_Calculus_(Feldman_Rechnitzer_and_Yeager)/01%3A_Vectors_and_Geometry_in_Two_and_Three_Dimensions/1.03%3A_Equations_of_Lines_in_2d

slope1 = eig1[1]/eig1[0]
line_y = numpy.polyval([-slope1, (slope1*mean_x)+mean_y], line_x)

ax.plot(
    line_x,
    line_y,
    "r--",
)

slope2 = eig2[1]/eig2[0]
line_y = numpy.polyval([-slope2, (slope2*mean_x)+mean_y], line_x)

ax.plot(
    line_x,
    line_y,
    "b--",
)

ax.set_xlim(line_x)
ax.set_ylim([min(y), max(y)])

plt.show(block = False)

print()