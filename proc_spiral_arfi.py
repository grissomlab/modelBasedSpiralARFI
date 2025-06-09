"""
proc_spiral_arfi.py
-------------------------

Load ARFI image data and run a model-based displacement map calculation.

Supporting Information Code to the manuscript:

Alternating-contrast single-shot spiral MR-ARFI with model-based displacement
map reconstruction

by S Sengupta et al, submitted to Magnetic Resonance in Medicine in 2025

"""

import torch
import numpy as np
import sigpy.plot as pl
import scipy.io as sio

data_dict = sio.loadmat("spiral20_data.mat")
gradStrength = data_dict["gradStrength"].item()  # mT/m
MEGdur = data_dict["MEGdur"].item()  # ms
rad_to_um = 1 / (2 * np.pi * 42.58 * 1e6 * gradStrength * 1e-3 * MEGdur * 1e-3) * 1e6  # rad to um conversion factor
images = data_dict["spiral20_data"].transpose(2, 3, 0, 1)
n_slices, n_images, Nx, Ny = images.shape

dataset = "spiral20"
disp_phase_polarity = np.ones((n_images,), dtype=float)
disp_phase_polarity[1::2] = -1
lamda = 2000
polynomial_order = 2
brain_row_start = 45  # brain boundaries
brain_row_end = 115
brain_col_start = 51
brain_col_end = 111
roi_row_start = 90  # roi boundaries
roi_row_end = 100
roi_col_start = 51
roi_col_end = 111
slices_to_recon = np.arange(n_slices)  # slices to reconstruct

images = torch.asarray(images)
images_crop = torch.zeros_like(images)
# crop the images to the brain
images_crop[..., brain_row_start:brain_row_end, brain_col_start:brain_col_end] = images[
    ..., brain_row_start:brain_row_end, brain_col_start:brain_col_end
]
images = images_crop.clone()
del images_crop

del data_dict

# create an ROI mask and apply phase correction
roi_mask = np.zeros((Nx, Ny), dtype=bool)
roi_mask[90:100, 70:90] = 1
for i in range(n_images):
    for j in range(n_slices):
        images[j, i] *= np.exp(-1j * np.angle(images[j, i][roi_mask].mean()))

# ROI Phase-corrected displacement map calculation
neg_image = images[:, disp_phase_polarity == -1, ...].mean(axis=1)
pos_image = images[:, disp_phase_polarity == 1, ...].mean(axis=1)
phsDiff = np.angle(pos_image * np.conj(neg_image))
roi_displacement = phsDiff * rad_to_um / 2.0


def calc_images(m0, poly_coeffs, A, disp_phase):

    # generate model images for comparison to data, for updating polynomial coefficients and displacement phase
    # m0: Nx x Ny m0 map
    # poly_coeffs: n_image x Npoly polynomial coefficients for each image
    # A: Nx*Ny x Npoly polynomial matrix
    # disp_phase: Nx x Ny displacement phase map

    n_image = poly_coeffs.shape[0]
    images = torch.zeros(((n_image,) + m0.shape), dtype=torch.complex128)
    for i in range(n_image):
        images[i, ...] = m0 * torch.exp(1j * (A @ poly_coeffs[i, :] + disp_phase_polarity[i].item() * disp_phase))

    return images


def update_m0(images, poly_coeffs, A, disp_phase):

    # m0 should be the average of the images, after removing the polynomial and displacement phases
    n_image = poly_coeffs.shape[0]
    m0 = torch.zeros((images.shape[1], images.shape[2]), dtype=torch.complex128)
    for i in range(n_image):
        m0 += images[i] * torch.exp(-1j * (A @ poly_coeffs[i, :] + disp_phase_polarity[i].item() * disp_phase))
    m0 /= n_image
    return m0


def worker():
    m0 = update_m0(images[si].detach(), poly_coeffs, A.detach(), disp_phase)
    images_pred = calc_images(m0, poly_coeffs, A.detach(), disp_phase)
    loss = torch.mean(torch.abs(images[si].detach() - images_pred) ** 2) + lamda * torch.mean(torch.abs(disp_phase))
    optimiser.zero_grad()
    loss.backward(retain_graph=False)
    print("loss: {}".format(loss.item()))
    return loss


# make the polynomial matrix
x, y = torch.meshgrid(torch.linspace(-1 / 2, 1 / 2, Nx), torch.linspace(-1 / 2, 1 / 2, Ny), indexing="ij")
A = torch.zeros((Nx, Ny, 1), dtype=float)
for i in range(polynomial_order + 1):
    for j in range(polynomial_order - i + 1):
        A = torch.cat((A, (x**i * y**j).reshape((Nx, Ny, 1))), dim=-1)
A = A[:, :, 1:]

displacement_model_based = np.zeros((n_slices, Nx, Ny), dtype=float)

for si in slices_to_recon:

    # initialize displacement phase and m0; init poly_coeffs to zeros
    disp_phase = torch.zeros((Nx, Ny), dtype=float)
    disp_phase.requires_grad = True
    poly_coeffs = torch.zeros((n_images, A.shape[-1]), dtype=float)
    poly_coeffs.requires_grad = True
    m0 = torch.zeros((Nx, Ny), dtype=torch.complex128)
    optimiser = torch.optim.LBFGS([poly_coeffs, disp_phase], lr=0.5, max_iter=200)
    optimiser.step(worker)

    displacement_model_based[si] = disp_phase.cpu().detach().numpy() * rad_to_um


pl.ImagePlot(
    displacement_model_based[slices_to_recon, ::-1, :],
    title="Model-Based Displacement (microns)",
    vmin=-0.2,
    vmax=0.2,
    colormap="plasma",
)
