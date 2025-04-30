import numpy as np
import torch

def add_dp_noise(z, epsilon=1.0, delta=1e-5, mechanism='laplace', mask=None):
    z_np = z.detach().cpu().numpy()
    sensitivity = 1.0

    if mechanism == 'laplace':
        scale = sensitivity / epsilon
        noise = np.random.laplace(0.0, scale, size=z_np.shape)
    elif mechanism == 'gaussian':
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        noise = np.random.normal(0.0, sigma, size=z_np.shape)
    else:
        raise ValueError("Unsupported mechanism.")

    if mask is not None:
        noise *= mask

    noise_tensor = torch.tensor(noise, dtype=z.dtype, device=z.device)
    return z + noise_tensor