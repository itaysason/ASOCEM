import os
import numpy as np
from src.utils import write_mrc, read_mrc, cryo_downsample
import numpy.linalg as linalg
import cv2
from skimage import measure
import matplotlib.pyplot as plt
import skimage
from tqdm import tqdm
import multiprocessing as mp
import warnings
import time
import numba as nb


warnings.filterwarnings("ignore")


class ASOCEMParams:
    def __init__(self, mgraph_path, output_dir, particle_size, downscale_size, area_size, contamination_criterion,
                 max_iter=600):
        self.mgraph_path = mgraph_path
        self.output_dir = output_dir
        self.particle_size = particle_size
        self.downscale_size = downscale_size
        self.area_size = area_size
        self.contamination_criterion = contamination_criterion
        self.max_iter = max_iter


def ASOCEM_ver1(micrograph_addr, output_dir, particle_size, downscale_size, area_size,
                contamination_criterion, max_iter=600, n_mgraphs_sim=10):

    # Require odd area_size, and odd downscale_size such that area_size | downscale_size
    area_size = area_size - 1 if area_size % 2 == 0 else area_size
    while downscale_size % area_size != 0 or downscale_size % 2 == 0:
        downscale_size -= 1

    micrograph_files = [os.path.join(micrograph_addr, a) for a in os.listdir(micrograph_addr) if '.mrc' in a]

    if len(micrograph_files) == 0:
        return

    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    # Prepare parameters
    all_params = []
    for mgraph_path in micrograph_files:
        all_params.append(ASOCEMParams(mgraph_path, output_dir, particle_size, downscale_size, area_size,
                                       contamination_criterion, max_iter=max_iter))

    # Execute everything

    if n_mgraphs_sim == 1:
        for params in tqdm(all_params):
            ASOCEM_one_mgraph_unpack(params)
    else:
        with mp.Pool(processes=n_mgraphs_sim) as pool:
            a = [0 for _ in pool.imap_unordered(ASOCEM_one_mgraph_unpack, all_params)]


def ASOCEM_one_mgraph_unpack(params):
    mgraph_path = params.mgraph_path
    output_dir = params.output_dir
    particle_size = params.particle_size
    downscale_size = params.downscale_size
    area_size = params.area_size
    contamination_criterion = params.contamination_criterion
    max_iter = params.max_iter
    ASOCEM_one_mgraph(mgraph_path, output_dir, particle_size, downscale_size, area_size, contamination_criterion,
                      max_iter=max_iter)


def ASOCEM_one_mgraph(mgraph_path, output_dir, particle_size, downscale_size, area_size, contamination_criterion,
                      max_iter=600):
    mgraph_name = os.path.split(mgraph_path)[1][:-4]  # Remove .mrc from file name
    curr_micrograph = read_mrc(mgraph_path).astype('float64')

    # Rescaling
    I = cryo_downsample(curr_micrograph, (downscale_size, downscale_size))

    phi = ASOCEM(I, area_size, contamination_criterion, max_iter)

    scaling_size = downscale_size / max(curr_micrograph.shape)
    d = max(3, int(np.ceil(scaling_size * particle_size / 8)))
    phi[:d] = 0
    phi[-d:] = 0
    phi[:, :d] = 0
    phi[:, -d:] = 0

    phi_seg = np.zeros(phi.shape)
    se_erod = max(area_size, int(np.ceil(scaling_size * particle_size / 6)))
    phi_erod = cv2.erode((phi > 0).astype('float'), np.ones((se_erod, se_erod), np.uint8))
    connected_components = measure.label(phi_erod)

    group_threshold = (scaling_size * 2 * particle_size) ** 2
    groups_id, group_sizes = np.unique(connected_components, return_counts=True)
    for group_id in groups_id:
        if np.sum(phi_erod[connected_components == group_id]) == 0:
            continue
        if np.count_nonzero(connected_components == group_id) > group_threshold:
            phi_seg[connected_components == group_id] = 1

    phi_seg = cv2.dilate(phi_seg, np.ones((se_erod, se_erod), np.uint8))

    phi_seg_big = cv2.resize(phi_seg, curr_micrograph.shape, interpolation=cv2.INTER_NEAREST)

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
    ax0.imshow(I, cmap='gray')
    ax0.axis('off')
    ax1.imshow(phi_seg, cmap='gray')
    ax1.axis('off')

    fig.savefig(os.path.join(output_dir, mgraph_name + '.png'))
    fig.clear(True)
    write_mrc(os.path.join(output_dir, mgraph_name + '_contaminations' + '.mrc'), phi_seg_big)


def ASOCEM(I, area_mat_sz, contamination_criterion, max_iter):
    dt = 10 ** 0
    mu = 10 ** -2
    nu = 0
    eta = 10 ** -8
    eps = 1
    tol = 10 ** -3

    cov_mat_sz = area_mat_sz ** 2

    # Initialize phi as Lipschitz circ
    size_x, size_y = I.shape
    x, y = np.meshgrid(np.arange(-(size_x // 2), size_x // 2 + 1), np.arange(-(size_y // 2), size_y // 2 + 1))
    phi_0 = (min(size_x, size_y) / 3) ** 2 - (x ** 2 + y ** 2)
    phi_0 /= np.max(np.abs(phi_0))

    # Chan Vese time process
    phi = chan_vesse_process(I, phi_0, cov_mat_sz, dt, mu, nu, eta, eps, max_iter, tol)

    if contamination_criterion == 'size':
        def est_func(group):
            return np.count_nonzero(group)
    elif contamination_criterion == 'power':
        def est_func(group):
            return np.mean(I[group])
    else:
        raise ValueError('contamination_criterion can only be size or power, got {} instead'.
                         format(contamination_criterion))

    group1_score = est_func(phi > 0)
    group2_score = est_func(phi < 0)
    if group1_score >= group2_score:
        phi *= -1
    return phi


def chan_vesse_process(I, phi_0, cov_mat_sz, dt, mu, nu, eta, eps, max_iter, tol):
    phi = neumann_bound_cond_mod(phi_0)
    size_0, size_1 = I.shape
    phi_old = phi.copy()
    area = int(np.sqrt(cov_mat_sz))

    # Precomputing all patches in image I
    I_patches = np.zeros((size_0 // area, size_1 // area, cov_mat_sz))
    for i in range(size_0 // area):
        for j in range(size_1 // area):
            I_patches[i, j] = I[i * area:(i + 1) * area, j * area:(j + 1) * area].flatten()

    I_patches_vector = np.reshape(I_patches, ((size_0 // area) * (size_1 // area), cov_mat_sz))

    # TODO: This criterion feels off, for example for area = 5 I get max_index = 193
    min_allowed_index_0 = 1
    max_allowed_index_0 = size_0 - size_0 % area - 1
    min_allowed_index_1 = 1
    max_allowed_index_1 = size_1 - size_1 % area - 1

    stop_condition_n = 5
    for iteration in range(1, max_iter + 1):
        phi_max = skimage.measure.block_reduce(phi, (area, area), np.max)
        phi_min = skimage.measure.block_reduce(phi, (area, area), np.min)

        patch_0 = I_patches[np.where(phi_max >= 0)].T.copy()
        patch_1 = I_patches[np.where(phi_min <= 0)].T.copy()

        # Compute mean and covariance
        mu0_est = np.mean(patch_0, 1)
        cov0_est = np.cov(patch_0)
        mu1_est = np.mean(patch_1, 1)
        cov1_est = np.cov(patch_1)

        # Micrograph is bad if any of these is true
        if patch_0.shape[1] <= 10 or patch_1.shape[1] <= 10:
            phi = np.ones(phi.shape)
            return phi

        cov0_inv = linalg.pinv(cov0_est)
        cov1_inv = linalg.pinv(cov1_est)
        logdet0 = logdet_amitay(cov0_est)
        logdet1 = logdet_amitay(cov1_est)

        # Compute phi near zero
        band_width = np.min(np.abs(phi[phi != 0]))
        median = np.median(np.abs(phi))
        # In the case log median / bw is whole number then bw will be larger than less than half
        # floor + 1 ensures more than half
        power = np.floor(np.log2(median / band_width)) + 1
        band_width *= 2 ** power

        row, col = np.where(np.logical_and(phi < band_width, phi > -band_width))

        good_indices = np.logical_and(np.logical_and(min_allowed_index_0 <= row, row < max_allowed_index_0),
                                      np.logical_and(min_allowed_index_1 <= col, col < max_allowed_index_1))
        row = row[good_indices]
        col = col[good_indices]

        sorted_indices = np.lexsort((row, col))
        row = row[sorted_indices]
        col = col[sorted_indices]

        # Compute rt for each f, techneclly we might not use some of these but it is probably faster to do for all
        x1 = I_patches_vector - mu1_est
        x0 = I_patches_vector - mu0_est
        rts = -nu + (logdet1 - logdet0 + multi_quadratic_form(x1, cov1_inv) - multi_quadratic_form(x0, cov0_inv)) / \
              (2 * cov_mat_sz)
        rts = np.reshape(rts, (size_0 // area, size_1 // area))
        rts = rts[row // area, col // area]

        # Compute all deltas and multiply by dt
        dt_deltas = dt * delta_eps(phi[row, col], eps)

        nb_update(phi, row, col, rts, dt_deltas, mu, eta)

        phi = neumann_bound_cond_mod(phi)

        # Stop criteria
        # TODO: Think about better stop criteria
        if iteration % stop_condition_n == 0 and iteration >= stop_condition_n * 2:
            area_new = phi > 0
            area_old = phi_old > 0
            changed = area_new != area_old
            if np.count_nonzero(changed) / np.count_nonzero(area_old) < tol:
                break
        if iteration % stop_condition_n == 0:
            phi_old = phi.copy()
    return phi


@nb.jit
def nb_update(phi, row, col, rts, dt_deltas, mu, eta):
    for i, j, rt, dt_delta in zip(row, col, rts, dt_deltas):
        phi[i, j] = time_ev_cov(phi[i - 1:i + 2, j - 1:j + 2], rt, dt_delta, mu, eta)


@nb.jit
def aij(phi_ij, phi_ip1j, phi_ijm1, phi_ijp1, mu, eta):
    return mu / np.sqrt(eta ** 2 + (phi_ip1j - phi_ij) ** 2 + ((phi_ijp1 - phi_ijm1) / 2) ** 2)


@nb.jit
def time_ev_cov(phi, rt, dt_delta_ij, mu, eta):
    a_im1j = aij(phi[0, 1], phi[1, 1], phi[0, 0], phi[0, 2], mu, eta)
    a_ij = aij(phi[1, 1], phi[2, 1], phi[1, 0], phi[1, 2], mu, eta)
    b_im1j = aij(phi[1, 0], phi[1, 1], phi[0, 0], phi[2, 0], mu, eta)
    b_ij = aij(phi[1, 1], phi[1, 2], phi[0, 1], phi[2, 1], mu, eta)

    cft = a_ij * phi[2, 1] + a_im1j * phi[0, 1] + b_ij * phi[1, 2] + b_im1j * phi[1, 0]

    return (phi[1, 1] + dt_delta_ij * (cft + rt)) / (1 + dt_delta_ij * (a_im1j + a_ij + b_im1j + b_ij))


def delta_eps(t, eps):
    return eps / (np.pi * (eps ** 2 + t ** 2))


def multi_quadratic_form(x, A):
    return np.sum(x @ A * x, 1)


def neumann_bound_cond_mod(f, h=1):
    g = f.copy()
    g[[0, -1, 0, -1], [0, 0, -1, -1]] = g[[h, -h - 1, h, -h - 1], [h, h, -h - 1, -h - 1]]
    g[[0, -1], 1:-1] = g[[h, -h - 1], 1:-1]
    g[1:-1, [0, -1]] = g[1:-1, [h, -h - 1]]
    return g


def logdet_amitay(mat):
    eig_vals, _ = linalg.eig(mat)
    eig_vals[eig_vals < 10e-8] = 1
    return np.sum(np.log(eig_vals))
