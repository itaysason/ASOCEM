import mrcfile
import numpy as np


def write_mrc(file_path, x):
    # For now it is transposed, when moving to C aligned this should be removed
    with mrcfile.new(file_path, overwrite=True) as mrc_fh:
        mrc_fh.set_data(x.astype('float32').T)
    return


def read_mrc(file_path):
    return np.ascontiguousarray(mrcfile.open(file_path).data.T)


def cryo_downsample(x, out_shape):
    """
    :param x: ndarray of size (N_1,...N_k)
    :param out_shape: iterable of integers of length k. The value in position i (n_i) is the size we want to cut from
        the center of x in dimension i. If the value of n_i <= 0 or >= N_i then the dimension is left as is.
    :return: out: downsampled x
    """
    dtype_in = x.dtype
    in_shape = np.array(x.shape)
    out_shape = np.array([s if 0 < s < in_shape[i] else in_shape[i] for i, s in enumerate(out_shape)])
    fourier_dims = np.array([i for i, s in enumerate(out_shape) if 0 < s < in_shape[i]])
    size_in = np.prod(in_shape[fourier_dims])
    size_out = np.prod(out_shape[fourier_dims])

    fx = crop(np.fft.fftshift(np.fft.fft2(x, axes=fourier_dims), axes=fourier_dims), out_shape)
    out = np.fft.ifft2(np.fft.ifftshift(fx, axes=fourier_dims), axes=fourier_dims) * (size_out / size_in)
    return out.astype(dtype_in)


def crop(x, out_shape):
    """
    :param x: ndarray of size (N_1,...N_k)
    :param out_shape: iterable of integers of length k. The value in position i is the size we want to cut from the
        center of x in dimension i. If the value is <= 0 then the dimension is left as is
    :return: out: The center of x with size outshape.
    """
    in_shape = np.array(x.shape)
    out_shape = np.array([s if s > 0 else in_shape[i] for i, s in enumerate(out_shape)])
    start_indices = in_shape // 2 - out_shape // 2
    end_indices = start_indices + out_shape
    indexer = tuple([slice(i, j) for (i, j) in zip(start_indices, end_indices)])
    out = x[indexer]
    return out
