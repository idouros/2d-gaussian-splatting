import os
from optparse import OptionParser
import configparser


def prepare_output_folder(folder):
    if os.path.exists(folder):
        if not os.path.isdir(folder):
            os.remove(folder)
            os.mkdir(folder)
    else:
        os.mkdir(folder)


def render():
    return False


def train(num_epochs, render_interval):
    for i in range(1,num_epochs+1):
        if i == 1 or i % render_interval == 0:
            print("Epoch {0}".format(i))
            render()


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
    render_interval = config.getint('training', 'render_interval')
    prepare_output_folder(output_folder)

    train(num_epochs, render_interval)

    print("Done!")

if __name__ == "__main__":
    main()