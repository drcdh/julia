import Image
import numpy as np

from math import log
from matplotlib.pyplot import get_cmap

width = 0.5
height = 0.5
center = complex(0,0)

xPixels = 1024
yPixels = 1024

maxIterations = 900
zLimit = 2.0

#C = complex(-0.70176, -0.3842)
C = complex(-0.4, 0.6)
#C = complex(-0.8, 0.156)

cmap = get_cmap('cubehelix')

def f(z):
    return pow(z,2) + C

def calc_p(final, penult):
    return log(abs(final))/log(abs(penult))

def iterate(zn, znm1=None, n=0):
    norm = abs(zn)
    if norm > zLimit:
        return norm, n
    if n >= maxIterations:
        return zLimit, maxIterations
    znp1 = f(zn)
    return iterate(znp1, zn, n+1)

def pixelCoord(i,j):
    px = (width/2.0)*float(i-xPixels/2)/float(xPixels/2)
    py = (height/2.0)*float(j-yPixels/2)/float(yPixels/2)
    pz = center + complex(px, py)
    return pz

def colorize(i, norm):
    c = float(i)/float(norm)
    r, g, b, l = cmap(c)
    return [int(255*x) for x in (b, g, r)]

if __name__ == '__main__':
    global largestNorm
    grid = np.ndarray(shape=(yPixels, xPixels), dtype=np.uint8)

    longestIter = 0.0

    for (j, i), v in np.ndenumerate(grid):
        pz = pixelCoord(i,j)
        finalMag, numIter = iterate(pz)
#        smoothedIter = numIter + 1.0 + log(log(zLimit)/log(finalMag))/log(2.0)
        smoothedIter = numIter
        if smoothedIter > longestIter:
            longestIter = smoothedIter
        grid[i, j] = smoothedIter

    print('Longest iteration: %d' % longestIter)

    image = np.ndarray(shape=(yPixels, xPixels, 3), dtype=np.uint8)

    for (i, j), v in np.ndenumerate(grid):
        image[i, j] = colorize(v, longestIter)

    img = Image.fromarray(image, 'RGB')
    img.save('julia.png')

