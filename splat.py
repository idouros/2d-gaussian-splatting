import os
from optparse import OptionParser
import configparser
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torchmetrics.image import StructuralSimilarityIndexMeasure


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

    return colour_samples / 255.0 , sample_coords / max_dim


def render(gaussians, image_shape):
    return False


def L1_and_SSIM(image_tensor_1, image_tensor_2, llambda):
    l1 = nn.L1Loss()
    L1 = l1(image_tensor_1, image_tensor_2)

    image_tensor_1_reshaped = image_tensor_1.unsqueeze(0).permute(0, 3, 1, 2)
    image_tensor_2_reshaped = image_tensor_2.unsqueeze(0).permute(0, 3, 1, 2)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    SSIM = ssim(image_tensor_1_reshaped, image_tensor_2_reshaped)

    combined = (1.0-llambda) * L1 + llambda * (1 - SSIM)
    return combined


def save_output_image(output_folder, output_image, epoch):
    output_file_name = os.path.join(output_folder, "epoch_{0}.png".format(str(epoch.zfill(5))))
    output_image.save(output_file_name)


def save_target_image(output_folder, target_image_array):
    output_file_name = os.path.join(output_folder, "target_image.png")
    target_image = Image.fromarray((target_image_array * 255).astype(np.uint8))
    target_image.save(output_file_name)


def train(input_image, target_image, num_samples, num_epochs, learning_rate, render_interval, output_folder):
    
    # take samples from the image (the gaussian means)
    colour_samples, sample_coords = sample_input_image(input_image, num_samples)
    colours = torch.tensor(colour_samples).float()
    coords = torch.atanh(torch.tensor(sample_coords).float())

    # variance and direction
    # Using this representation for training instead of a covariance matrix because:
    # a. it is more immediately intuitive, and
    # b. coveriance matrix would need to be positive semidefinite, which is not that easy to constrain
    variances = torch.rand(num_samples, 2)
    directions = 2.0 * torch.rand(num_samples, 1) - 1 # In 3D it's a quaternion or a rotation matrix, in 2D an angle is enough

    # finally, the alpha values for the blending
    alphas = torch.ones(num_samples, 1)

    # We now have all the params for optimizing
    Y = nn.Parameter(torch.cat([coords, variances, directions, colours, alphas], dim = 1))
    optimizer = Adam([Y], lr = learning_rate) 

    # TODO NEXT : Implement the rendering before continuing with this 
    for i in range(1,num_epochs+1):
        #output_image = render(Y, target_image.shape)
        output_image_tensor = torch.tensor(target_image, requires_grad=True)
        target_image_tensor = torch.tensor(target_image, requires_grad=True)
        loss = L1_and_SSIM(output_image_tensor, target_image_tensor, 0.2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i == 1 or i % render_interval == 0:
            print("Epoch {0}, loss {1}".format(i, loss.item()))
            #save_output_image(output_folder, Image.fromarray(output_image.astype(np.uint8)), i)


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
    prepare_output_folder(output_folder)

    input_image = Image.open(input_file)
    target_image = prepare_target_image(input_image, target_image_height, target_image_width)
    save_target_image(output_folder, target_image)
    train(input_image, target_image, num_samples, num_epochs, learning_rate, render_interval, output_folder)

    print("Done!")

if __name__ == "__main__":
    main()