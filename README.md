# step-detect

Python algorithms for autonomous step detection in 1D data. Specifically designed for single molecule force spectroscopy, but applicable to any cases in which there are sudden steps in otherwise semi-stable data.

This module contains functions for transforming raw data into a signal that is suitable for step detection. In basic terms, this amounts to edge-preserving smoothing, though in practice the calculation of these methods is slightly more complex. Once the signal is obtained, a simple thresholding algorithm is applied to determine the positions of steps and their relative magnitudes.

The signal-generating functions follow two distinct methods:

1. Multiscale product as described by [Mallat and Zhong](http://www.cmap.polytechnique.fr/~mallat/papiers/MallatEdgeCharact92.pdf), and as implemented by [Sadler and Swami](http://www.dtic.mil/dtic/tr/fulltext/u2/a351960.pdf).
2. The t-statistic method of [Carter and Cross](http://www.nature.com/nature/journal/v435/n7040/abs/nature03528.html).

The usage of this module is demonstrated in the supplied demo IPython notebook, which also provides comparision between the above two methods as well as convolution with a derivative-of-Gaussian wavelet (a simpler but less robust method for step detection).
