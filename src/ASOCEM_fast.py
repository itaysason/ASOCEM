import os
import numpy as np
from src.utils import write_mrc, read_mrc, cryo_downsample
import numpy.linalg as linalg
import cv2
from skimage import measure
import matplotlib.pyplot as plt
import skimage
import warnings
import shutil
import random


warnings.filterwarnings("ignore")


class DataHolder:
    def __init__(self, tmp_dir, area, n_mgraphs, stop_condition_n, max_iter, contamination_criterion):
        self.tmp_dir = tmp_dir
        self.mgraphs_queue = [os.path.join(tmp_dir, a) for a in os.listdir(tmp_dir) if '_ds_mgraph.npy' in a]
        random.shuffle(self.mgraphs_queue) # Shuffle input randomly
        self.n_mgraphs = min(n_mgraphs, len(self.mgraphs_queue))
        self.stop_condition_n = stop_condition_n
        self.max_iter = max_iter
        self.area = int(area)
        self.cov_mat_size = area * area
        self.contamination_criterion = contamination_criterion

        # Find longest name
        longest_name = max([len(os.path.split(mgraph_path)[1][:-4]) for mgraph_path in self.mgraphs_queue])

        # Find shape of I and computing phi_0
        some_I = np.load(self.mgraphs_queue[0])
        self.size_x, self.size_y = some_I.shape
        self.phi_0 = initialize_phi_0(self.size_x, self.size_y)

        # Defining batches of data
        self.Is = np.empty((self.n_mgraphs, self.size_x, self.size_y))
        self.I_patches = np.empty((self.n_mgraphs, self.size_x // area, self.size_y // area, self.cov_mat_size))
        self.I_patches_vector = np.empty(
            (self.n_mgraphs, (self.size_x // area) * (self.size_y // area), self.cov_mat_size))
        self.phis = np.empty((self.n_mgraphs, self.size_x, self.size_y))
        self.old_phis = np.empty((self.n_mgraphs, self.size_x, self.size_y))
        self.names = np.array(['empty' * (longest_name // 5 + 1)] * self.n_mgraphs)
        self.iterations = np.empty(self.n_mgraphs, dtype='int')
        self.all_mu0_est = np.empty((self.n_mgraphs, self.cov_mat_size))
        self.all_mu1_est = np.empty((self.n_mgraphs, self.cov_mat_size))
        self.all_cov0_inv = np.empty((self.n_mgraphs, self.cov_mat_size, self.cov_mat_size))
        self.all_cov1_inv = np.empty((self.n_mgraphs, self.cov_mat_size, self.cov_mat_size))
        self.all_logdet0 = np.empty(self.n_mgraphs)
        self.all_logdet1 = np.empty(self.n_mgraphs)

        # Initializing everything
        for i in range(self.n_mgraphs):
            self.next(i)

    def next(self, i):
        if 'empty' not in self.names[i]:
            # Save
            phi = self.phis[i].copy()
            I = self.Is[i]
            if self.contamination_criterion == 'size':
                def est_func(group):
                    return np.count_nonzero(group)
            elif self.contamination_criterion == 'power':
                def est_func(group):
                    return np.mean(I[group])
            else:
                raise ValueError('contamination_criterion can only be size or power, got {} instead'.
                                 format(self.contamination_criterion))

            group1_score = est_func(phi > 0)
            group2_score = est_func(phi < 0)
            if group1_score >= group2_score:
                phi *= -1
            np.save(os.path.join(self.tmp_dir, self.names[i] + '_phi.npy'), phi)

        if len(self.mgraphs_queue) == 0:
            # If no more are left in queue simply remove it
            keep = np.where(np.arange(self.n_mgraphs) != i)
            self.Is = self.Is[keep]
            self.I_patches = self.I_patches[keep]
            self.I_patches_vector = self.I_patches_vector[keep]
            self.phis = self.phis[keep]
            self.old_phis = self.old_phis[keep]
            self.names = self.names[keep]
            self.iterations = self.iterations[keep]
            self.all_mu0_est = self.all_mu0_est[keep]
            self.all_mu1_est = self.all_mu1_est[keep]
            self.all_cov0_inv = self.all_cov0_inv[keep]
            self.all_cov1_inv = self.all_cov1_inv[keep]
            self.all_logdet0 = self.all_logdet0[keep]
            self.all_logdet1 = self.all_logdet1[keep]
            self.n_mgraphs -= 1
        else:
            # If there are still mgraphs to handle, insert it instead of i
            mgraph_path = self.mgraphs_queue.pop()
            self.Is[i] = np.load(mgraph_path)
            size_x, size_y, area, cov_mat_size = self.size_x, self.size_y, self.area, self.cov_mat_size
            I_patches = self.I_patches[i]
            for x in range(size_x // area):
                for y in range(size_y // area):
                    I_patches[x, y] = self.Is[i, x * area:(x + 1) * area, y * area:(y + 1) * area].flatten()

            self.I_patches_vector[i] = np.reshape(I_patches, ((size_x // area) * (size_y // area), cov_mat_size))
            self.phis[i] = self.phi_0.copy()
            self.old_phis[i] = self.phi_0.copy()
            self.names[i] = os.path.split(mgraph_path)[1][:-4]
            self.iterations[i] = 0

    def compute_statistics(self):
        phis_max = skimage.measure.block_reduce(self.phis, (1, self.area, self.area), np.max)
        phis_min = skimage.measure.block_reduce(self.phis, (1, self.area, self.area), np.min)

        for i, (phi_max, phi_min) in enumerate(zip(phis_max, phis_min)):
            patch_0 = self.I_patches[i][np.where(phi_max >= 0)].T
            patch_1 = self.I_patches[i][np.where(phi_min <= 0)].T

            # Compute mean and covariance
            self.all_mu0_est[i] = np.mean(patch_0, 1)
            cov0_est = np.cov(patch_0)
            self.all_mu1_est[i] = np.mean(patch_1, 1)
            cov1_est = np.cov(patch_1)

            self.all_cov0_inv[i] = linalg.pinv(cov0_est)
            self.all_cov1_inv[i] = linalg.pinv(cov1_est)
            self.all_logdet0[i] = logdet_amitay(cov0_est)
            self.all_logdet1[i] = logdet_amitay(cov1_est)

    def get_all_rts(self):
        x0 = self.I_patches_vector - self.all_mu0_est[:, np.newaxis, :]
        x1 = self.I_patches_vector - self.all_mu1_est[:, np.newaxis, :]
        tmp_logdet = self.all_logdet1[:, np.newaxis] - self.all_logdet0[:, np.newaxis]
        tmp_qf = multi_quadratic_form(x1, self.all_cov1_inv) - multi_quadratic_form(x0, self.all_cov0_inv)
        rts = (tmp_logdet + tmp_qf) / (2 * self.cov_mat_size)
        return np.reshape(rts, (self.n_mgraphs, self.size_x // self.area, self.size_y // self.area))

    def step(self, nu, dt, eps):
        rts = -nu + self.get_all_rts()

        dt_deltas = dt * delta_eps(self.phis[:, ::self.area, ::self.area], eps)
        dt_deltas *= rts
        for i in range(self.area):
            for j in range(self.area):
                self.phis[:, i::self.area, j::self.area] += dt_deltas

    def neumann_bound_cond_mod(self, h=1):
        self.phis[:, [0, -1, 0, -1], [0, 0, -1, -1]] = self.phis[:, [h, -h - 1, h, -h - 1], [h, h, -h - 1, -h - 1]]
        self.phis[:, [0, -1], 1:-1] = self.phis[:, [h, -h - 1], 1:-1]
        self.phis[:, 1:-1, [0, -1]] = self.phis[:, 1:-1, [h, -h - 1]]

    def check_stop_criterion(self, tol):
        # TODO: Think about better stop criteria
        done_indices = []
        for i in range(self.n_mgraphs):
            iteration = self.iterations[i]
            if iteration % self.stop_condition_n == 0:
                area_new = self.phis[i] > 0
                area_old = self.old_phis[i] > 0
                changed = area_new != area_old
                if np.count_nonzero(changed) / np.count_nonzero(area_old) < tol or iteration >= self.max_iter:
                    print(self.names[i], iteration)
                    done_indices.append(i)
                self.old_phis[i] = self.phis[i].copy()

        if len(done_indices) > 0:
            done_indices = -np.sort(-np.array(done_indices))
            for i in done_indices:
                self.next(i)


def ASOCEM_ver1(micrograph_addr, output_dir, particle_size, downscale_size, area_size,
                contamination_criterion, tmp_dir='./tmp', n_mgraphs_sim=10):

    # Require odd area_size, and odd downscale_size such that area_size | downscale_size
    area_size = area_size - 1 if area_size % 2 == 0 else area_size
    while downscale_size % area_size != 0 or downscale_size % 2 == 0:
        downscale_size -= 1

    micrograph_files = [os.path.join(micrograph_addr, a) for a in os.listdir(micrograph_addr) if '.mrc' in a]

    # If there are no mrc files
    if len(micrograph_files) == 0:
        return

    # Create output_dir if needed
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    # Create tmp_dir if needed
    try:
        os.makedirs(tmp_dir)
    except OSError:
        pass

    # Preprocessing - Saving all downsampled micrographs
    mgraph_names = []
    mgraph_shapes = []
    for mgraph_path in micrograph_files:
        mgraph_name = os.path.split(mgraph_path)[1][:-4]  # Remove .mrc from file name
        mgraph_names.append(mgraph_name)
        curr_micrograph = read_mrc(mgraph_path).astype('float64')
        mgraph_shapes.append(curr_micrograph.shape)

        # Rescaling
        I = cryo_downsample(curr_micrograph, (downscale_size, downscale_size))
        np.save(os.path.join(tmp_dir, mgraph_name + '_ds_mgraph.npy'), I)

    # Executing ASOCEM
    ASOCEM(tmp_dir, area_size, contamination_criterion, n_mgraphs_sim=n_mgraphs_sim)

    # Post processing
    for mgraph_name, mgraph_shape in zip(mgraph_names, mgraph_shapes):
        I = np.load(os.path.join(tmp_dir, mgraph_name + '_ds_mgraph.npy'))
        phi = np.load(os.path.join(tmp_dir, mgraph_name + '_ds_mgraph_phi.npy'))

        scaling_size = downscale_size / max(mgraph_shape)
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
        group_ids, group_sizes = np.unique(connected_components, return_counts=True)
        indices = np.argsort(-group_sizes)
        group_ids, group_sizes = group_ids[indices], group_sizes[indices]
        tmp = np.full(connected_components.shape, False)
        for group_id, group_size in zip(group_ids, group_sizes):
            if np.sum(phi_erod[connected_components == group_id]) == 0:
                continue
            tmp = np.logical_or(tmp, connected_components == group_id)
            if group_size > group_threshold:
                phi_seg[connected_components == group_id] = 1

        phi_seg = cv2.dilate(phi_seg, np.ones((se_erod, se_erod), np.uint8))

        phi_seg_big = cv2.resize(phi_seg, mgraph_shape, interpolation=cv2.INTER_NEAREST)

        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
        ax0.imshow(I, cmap='gray')
        ax0.axis('off')
        ax1.imshow(phi_seg, cmap='gray')
        ax1.axis('off')

        fig.savefig(os.path.join(output_dir, mgraph_name + '.png'))
        fig.clear(True)
        write_mrc(os.path.join(output_dir, mgraph_name + '_contaminations' + '.mrc'), phi_seg_big)

    # Delete tmp_dir
    try:
        shutil.rmtree(tmp_dir)
    except OSError:
        pass


def ASOCEM(tmp_dir, area, contamination_criterion, n_mgraphs_sim=10):
    dt = 10 ** 0
    nu = 0
    eps = 1
    tol = 10 ** -3
    max_iter = 600
    area = int(area)

    # Chan Vesse time process
    chan_vesse_process_many(tmp_dir, area, dt, nu, eps, max_iter, tol, contamination_criterion, n_mgraphs_sim)


def chan_vesse_process_many(tmp_dir, area, dt, nu, eps, max_iter, tol, contamination_criterion, n_mgraphs=10):
    stop_condition_n = 5
    data_holder = DataHolder(tmp_dir, area, n_mgraphs, stop_condition_n, max_iter, contamination_criterion)

    while data_holder.n_mgraphs > 0:
        # Updating iterations
        data_holder.iterations += 1

        # Computing statistics
        data_holder.compute_statistics()

        # Step
        data_holder.step(nu, dt, eps)

        # Neumann bound
        data_holder.neumann_bound_cond_mod()

        # Stop criteria
        data_holder.check_stop_criterion(tol)


def initialize_phi_0(size_x, size_y):
    x, y = np.meshgrid(np.arange(-(size_x // 2), size_x // 2 + 1), np.arange(-(size_y // 2), size_y // 2 + 1))
    phi_0 = (min(size_x, size_y) / 3) ** 2 - (x ** 2 + y ** 2)
    phi_0 /= np.max(np.abs(phi_0))
    return neumann_bound_cond_mod(phi_0)


def delta_eps(t, eps):
    return eps / (np.pi * (eps ** 2 + t ** 2))


def multi_quadratic_form(x, A):
    return np.sum(x @ A * x, -1)


def neumann_bound_cond_mod(f, h=1):
    g = f.copy()
    g[[0, -1, 0, -1], [0, 0, -1, -1]] = g[[h, -h - 1, h, -h - 1], [h, h, -h - 1, -h - 1]]
    g[[0, -1], 1:-1] = g[[h, -h - 1], 1:-1]
    g[1:-1, [0, -1]] = g[1:-1, [h, -h - 1]]
    return g


def neumann_bound_cond_mod_many(f, h=1):
    g = f.copy()
    g[:, [0, -1, 0, -1], [0, 0, -1, -1]] = g[:, [h, -h - 1, h, -h - 1], [h, h, -h - 1, -h - 1]]
    g[:, [0, -1], 1:-1] = g[:, [h, -h - 1], 1:-1]
    g[:, 1:-1, [0, -1]] = g[:, 1:-1, [h, -h - 1]]
    return g


def logdet_amitay(mat):
    eig_vals, _ = linalg.eig(mat)
    eig_vals[eig_vals < 10e-8] = 1
    return np.sum(np.log(eig_vals))
