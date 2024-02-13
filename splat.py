import os
from optparse import OptionParser
import configparser
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torchvision.transforms as tvt
import math


def prepare_output_folder(folder):
    if os.path.exists(folder):
        if not os.path.isdir(folder):
            os.remove(folder)
            os.mkdir(folder)
    else:
        os.mkdir(folder)


def prepare_target_image(input_image, target_image_width, target_image_height):
    return np.array(input_image.resize([target_image_width, target_image_height]).convert('RGB')) / 255.0


def sample_input_image(input_image, num_samples):
    input_image = np.array(input_image.convert('RGB'))
    (input_height, input_width, colour_channels) = input_image.shape

    sample_coords_x = np.random.randint(0, input_width, size=(num_samples, 1))
    sample_coords_y = np.random.randint(0, input_height, size=(num_samples, 1))
    max_dim = max(input_height, input_width)
    sample_coords = np.hstack([sample_coords_x, sample_coords_y])

    colour_samples = np.empty((num_samples, colour_channels))
    for i in range(num_samples):
        colour_samples[i:] = input_image[sample_coords[i,0], sample_coords[i,1]]

    return colour_samples / 255.0 , ((sample_coords / max_dim) - 0.5) * 2.0


def L1_and_SSIM(image_tensor_1, image_tensor_2, llambda):
    l1 = nn.L1Loss()
    L1 = l1(image_tensor_1, image_tensor_2)

    image_tensor_1_reshaped = image_tensor_1.unsqueeze(0)
    image_tensor_2_reshaped = image_tensor_2.unsqueeze(0)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    SSIM = ssim(image_tensor_1_reshaped, image_tensor_2_reshaped)

    combined = (1.0-llambda) * L1 + llambda * (1 - SSIM)
    return combined


def save_output_image(output_folder, output_image, epoch):
    output_file_name = os.path.join(output_folder, "epoch_{0}.png".format(str(epoch).zfill(5)))
    output_image.save(output_file_name)


def save_target_image(output_folder, target_image_array):
    output_file_name = os.path.join(output_folder, "target_image.png")
    target_image = Image.fromarray((target_image_array * 255).astype(np.uint8))
    target_image.save(output_file_name)


def render(means, variances, directions, colours, alphas, image_shape, gaussian_kernel_size = 10):

    nx, ny = (image_shape[0], image_shape[1])

    # reconstruct the covariance matrices
    # See here: https://en.wikipedia.org/wiki/Gaussian_function#Meaning_of_parameters_for_the_general_equation
    f = (0.5 / nx) * gaussian_kernel_size
    sx2 = variances[:,0] * variances[:,0] * f
    sy2 = variances[:,1] * variances[:,1] * f
    cosrho = torch.cos(directions)
    sinrho = torch.sin(directions)
    sin2rho = torch.sin(directions * 2)
    cosrho_sq = cosrho * cosrho
    sinrho_sq = sinrho * sinrho
    a = cosrho_sq/(2*sx2) + sinrho_sq/(2*sy2)
    b = sin2rho/(4*sx2)   + sin2rho/(4*sy2)
    c = sinrho_sq/(2*sx2) + cosrho_sq/(2*sy2)

    num_blobs = means.shape[0]
    x_step = 2.0 / nx
    y_step = 2.0 / ny
    x = torch.arange(-1.0, 1.0, step = x_step)
    y = torch.arange(-1.0, 1.0, step = y_step)
    combined_image = torch.zeros(3, nx, ny, requires_grad=True)

    # render each blob as a separate image, then add
    for k in range(0, num_blobs):
        constituent_image = torch.zeros(3, nx, ny, requires_grad=True)
        mx = means[k,0]
        my = means[k,1]
        colour = colours[k,:]
        alpha = alphas[k]
        i, j = torch.meshgrid(x, y, indexing='ij')
        # https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
        G = torch.exp(-(a[k]*(i-mx)*(i-mx) + b[k]*(i-mx)*(j-my) + c[k]*(j-my)*(j-my)))
        constituent_image = torch.stack([G*colour[0],G*colour[1],G*colour[2]])
        combined_image = combined_image + constituent_image * alpha

    # clamp values and return
    return torch.clamp(combined_image, 0, 1)


def train(input_image, target_image, gaussian_kernel_size, num_samples, num_epochs, learning_rate, density_control_params, render_interval, output_folder):

    torch.cuda.device(torch.cuda if torch.cuda.is_available() else 0)

    # take samples from the image (the gaussian means)
    colour_samples, sample_coords = sample_input_image(input_image, num_samples)
    colours = torch.tensor(colour_samples).float().requires_grad_()
    coords = torch.atanh(torch.tensor(sample_coords).float(), ).requires_grad_()

    # variance and direction
    # Using this representation for training instead of a covariance matrix because:
    # a. it is more immediately intuitive, and
    # b. coveriance matrix would need to be positive semidefinite, which is not that easy to constrain
    variances = torch.rand(num_samples, 2).requires_grad_()
    directions = (2.0 * torch.rand(num_samples, 1) - 1.0).requires_grad_() # In 3D it's a quaternion or a rotation matrix, in 2D an angle is enough

    # finally, the alpha values for the blending
    alphas = torch.ones(num_samples, 1)

    # We now have all the params for optimizing
    params = torch.cat([coords, variances, directions, colours, alphas], dim = 1)
    params.requires_grad_()
    Y = nn.Parameter(params, requires_grad=True)
    optimizer = Adam([Y], lr = learning_rate) 
    num_added_samples = 0
    for i in range(1,num_epochs+1):
        means = torch.tanh(Y[:, 0:2])
        variances = torch.sigmoid(Y[:, 2:4])
        directions = Y[:, 4] * math.pi
        colours = torch.sigmoid(Y[:, 5:8])
        alphas = torch.sigmoid(Y[:, 8])

        output_image = render(means, variances, directions, colours, alphas, target_image.shape, gaussian_kernel_size) 
        output_image_tensor = output_image.to(torch.float64)
        target_image_tensor = torch.tensor(target_image, requires_grad=True).permute(2, 0, 1)

        loss = L1_and_SSIM(output_image_tensor, target_image_tensor, 0.2)
        if i == 1 or i % render_interval == 0:
            print("Epoch {0}, loss {1}".format(i, loss.item()))
            save_output_image(output_folder, tvt.ToPILImage()(output_image), i)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Adaptive densification 
        if density_control_params['densify'] == True:
            if i % density_control_params['interval'] == 0:
                print("Densifying...")
                num_extra_samples = density_control_params['extra_samples']
                if num_added_samples < num_extra_samples:

                    # Almost-transparent gaussians (remove)
                    num_kernels_before = params.shape[0]
                    transparency_threshold = density_control_params['transparency_threshold']
                    mask = torch.sigmoid(Y[:, 8]) > transparency_threshold
                    params = params[mask]
                    grads = Y.grad[mask][:,0:2] # will need this a little bit later
                    num_kernels_after = params.shape[0]
                    print("\t{0} near-transparent kernels removed".format(num_kernels_before-num_kernels_after))

                    # Identify the kernels with large position gradients
                    grad_norms = torch.norm(grads, dim=1, p=2)
                    large_grad_mask = grad_norms > density_control_params['kernel_position_gradient_threshold']
                    
                    # Construct the variance norms
                    variances = torch.sigmoid(Y[:, 2:4])
                    var_norms = torch.norm(variances, dim=1, p=2)

                    # Large coordinate gradient - small Gaussian (indicates an under-reconstructed region):
                    # Create a clone of the gaussian, and move it in the direction of the gradient
                    mask_under_reconstructed = torch.logical_and(large_grad_mask, var_norms < density_control_params['kernel_size_threshold'])
                    kernels_cloned = 0
                    for k in range (0, mask_under_reconstructed.shape[0]):
                        if mask_under_reconstructed[k] == True:
                            cloned_kernel = torch.clone(params[k,:])
                            centre = cloned_kernel[0:2]
                            gradient = grads[k]
                            new_centre = centre + gradient
                            cloned_kernel[0:2] = new_centre
                            params = torch.cat([params, torch.reshape(cloned_kernel, (1,9))], dim=0)
                            kernels_cloned +=1
                    print("\t{0} kernels cloned to densify under-reconstructed regions".format(kernels_cloned))

                    # Large coordinate gradient - large Gaussian (indicates an over-reconstructed region):
                    # Split the gaussian: Create a copy, scale both down
                    kernels_split = 0
                    mask_over_reconstructed = torch.logical_and(large_grad_mask, var_norms >= density_control_params['kernel_size_threshold'])
                    for k in range (0, mask_over_reconstructed.shape[0]):
                        if mask_over_reconstructed[k] == True:
                            kernel_1 = torch.clone(params[k,:])
                            kernel_2 = torch.clone(params[k,:])
                            new_variance = variances[k,:] / density_control_params['kernel_scaledown_factor']
                            kernel_1[2:4] = new_variance
                            params[k,:] = kernel_1
                            kernel_2[2:4] = new_variance
                            params = torch.cat([params, torch.reshape(kernel_2, (1,9))], dim=0)
                            kernels_split +=1
                    print("\t{0} kernels split and scaled down to densify over-reconstructed regions".format(kernels_split))

                    # Reset the optimizer
                    Y = nn.Parameter(params, requires_grad=True)
                    optimizer = Adam([Y], lr = learning_rate) 



def main():

    # Read the configuration
    parser = OptionParser()
    (options, args) = parser.parse_args()
    if len(args) != 1:
        parser.error("Incorrect number of arguments. Please specify (only) a path to a config file")
        exit()
    config = configparser.ConfigParser()
    config_filename = args[0]
    config.read(config_filename)

    # Filename parameters
    input_file = config.get('files', 'input_file')
    output_folder = config.get('files', 'output_folder')
    num_epochs = config.getint('training', 'num_epochs')
    num_samples = config.getint('training', 'num_samples')
    learning_rate = config.getfloat('training', 'learning_rate')
    render_interval = config.getint('training', 'render_interval')
    target_image_height = config.getint('training', 'target_image_height') 
    target_image_width = config.getint('training', 'target_image_width')
    gaussian_kernel_size = config.getint('rendering', 'gaussian_kernel_radius', fallback=5) * 2
    density_control_params = {}
    density_control_params["densify"] = config.getboolean('density_control', 'densify', fallback=False)
    density_control_params["interval"] = config.getint('density_control', 'interval', fallback=0)
    density_control_params["extra_samples"] = config.getint('density_control', 'extra_samples', fallback=0)
    density_control_params["transparency_threshold"] = config.getfloat('density_control', 'transparency_threshold', fallback=0.005)
    density_control_params["kernel_size_threshold"] = config.getfloat('density_control', 'kernel_size_threshold', fallback = 0.0015)
    density_control_params["kernel_position_gradient_threshold"] = config.getfloat('density_control', 'kernel_position_gradient_threshold', fallback = 0.0015)
    density_control_params["kernel_scaledown_factor"] = config.getfloat('density_control', 'kernel_scaledown_factor', fallback = 1.6)
    prepare_output_folder(output_folder)

    # Load the image and prepare for training
    input_image = Image.open(input_file)
    target_image = prepare_target_image(input_image, target_image_height, target_image_width)
    save_target_image(output_folder, target_image)
    
    # The main job
    train(input_image, target_image, gaussian_kernel_size, num_samples, num_epochs, learning_rate, density_control_params, render_interval, output_folder)

    print("Done!")

if __name__ == "__main__":
    main()