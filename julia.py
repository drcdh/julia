"""Julia.

Usage:
    julia.py [--imgname=<filename>] [--imgsize=<px>] [--zoom=<z>] [--zlimit=<zlim>] [--center=<c_r>,<c_i>] [--cmap=<cmap>] [--soften] -- <z_r> <z_i>
    julia.py (-h | --help)

Options:
    -h --help               Show this screen.
    --imgname=<filename>    Pathname to write image [default: julia.png].
    --imgsize=<px>          Image size in pixels [default: 512].
    --zoom=<z>              Zoom on center (>1 is "in") [default: 2.0].
    --zlimit=<zlim>         Maximum value of |z| in recusion [default: 2.0].
    --center=<c_r>,<c_i>    Real and imaginary parts of center point [default: 0.0,0.0].
    --cmap=<cmap>           Colormap name [default: cubehelix].
    --soften                Soften the iterator.

"""

from docopt import docopt

import Image
import numpy as np

from math import log
from matplotlib.pyplot import get_cmap

maxIterations = 900

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
    px = (1.0/(2.0*zoom))*float(j-imgsize/2)/float(imgsize/2)
    py = (1.0/(2.0*zoom))*float(i-imgsize/2)/float(imgsize/2)
    pz = center + complex(px, py)
    return pz

def colorize(i, norm, cmap):
    c = float(i)/float(norm)
    r, g, b, l = cmap(c)
    return [int(255*x) for x in (b, g, r)]

if __name__ == '__main__':
    global C, imgsize, center, zoom, zLimit

    arg = docopt(__doc__, options_first=True)
    C = complex(*map(float, [arg['<z_r>'], arg['<z_i>']]))
    imgsize = int(arg['--imgsize'])
    center = complex(*map(float, arg['--center'].split(',')))
    zoom = float(arg['--zoom'])
    zLimit = float(arg['--zlimit'])
    cmap = get_cmap(arg['--cmap'])

    grid = np.ndarray(shape=(imgsize, imgsize), dtype=np.uint8)

    longestIter = 0.0

    for (j, i), v in np.ndenumerate(grid):
        pz = pixelCoord(i,j)
        finalMag, numIter = iterate(pz)
        smoothedIter = numIter
        if arg['--soften']:
            smoothedIter += 1.0 + log(log(zLimit)/log(finalMag))/log(2.0)
        if smoothedIter > longestIter:
            longestIter = smoothedIter
        grid[i, j] = smoothedIter

    print('Longest iteration: %d' % longestIter)

    image = np.ndarray(shape=(imgsize, imgsize, 3), dtype=np.uint8)

    for (i, j), v in np.ndenumerate(grid):
        image[i, j] = colorize(v, longestIter, cmap)

    img = Image.fromarray(image, 'RGB')
    img.save(arg['--imgname'])

