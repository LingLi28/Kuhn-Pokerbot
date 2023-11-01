import os
import random
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Current file marks the root directory
TRAINING_IMAGE_DIR = os.path.join(ROOT_DIR, "data_sets", "training_images")  # Directory for storing training images
TEST_IMAGE_DIR = os.path.join(ROOT_DIR, "data_sets", "test_images")  # Directory for storing test images
CODE_TEST_IMAGE_DIR = os.path.join(ROOT_DIR, "data_sets", "code_test_images")
LABELS = ['J', 'Q', 'K']  # Possible card labels
IMAGE_SIZE = 32 
ROTATE_MAX_ANGLE = 15

FONTS = [
    font_manager.findfont(font_manager.FontProperties(family = 'sans-serif', style = 'normal', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'sans-serif', style = 'italic', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'sans-serif', style = 'normal', weight = 'medium')),
    font_manager.findfont(font_manager.FontProperties(family = 'serif', style = 'normal', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'serif', style = 'italic', weight = 'normal')),
    font_manager.findfont(font_manager.FontProperties(family = 'serif', style = 'normal', weight = 'medium')),
]  # True type system fonts


def normalize_image(raw_image: Image):
    """
    Normalize a raw image to serve as input to the image classifier.

    Arguments
    ---------
    raw_image : Image
        Raw image to normalize.

    Returns
    -------
    image : matrix of floats between zero and one
        Normalized image that can be used by the image classifier.
    """

    # Get the pixels of the image in a list
    pixels_image = list(raw_image.getdata())

    # Get the maximum pixel value and the dimensions of the image
    maximum = max(pixels_image)
    width, height = raw_image.size

    # Normalize the image and return the image in the same shape as the original image
    image_list = [pixels_image[i] / maximum for i in range(height*width)]
    image = [image_list[i * width:(i + 1) * width] for i in range(height)]
    return image


def load_data_set(data_dir, n_validation = 0):
    """
    Normalize the images in data_dir and divide in a training and validation set.

    Parameters
    ----------
    data_dir : str
        Directory of images to load
    n_validation : int
        Number of images that are assigned to the validation set
    
    Returns
        -------
    training_images : list of matrices
        List of normalized training images.
    training_labels : list of strings
        List of labels that belong to the normalized training images.
    validation_images : list of matrices
        List of normalized validation images.
    validation_labels : list of strings
        List of labels that belong to the normalized validation images.
    """

    # Extract png files
    files = os.listdir(data_dir)
    png_files = []
    for file in files:
        if file.split('.')[-1] == "png":
            png_files.append(file)

    random.shuffle(png_files)  # Shuffled list of the png-file names that are stored in data_dir

    # Normalize the images, and store the normalized images and the labels in variables
    normalized_images = [0]*len(files)
    labels = []
    for i in range(len(png_files)):
        normalized_images[i] = normalize_image(Image.open(data_dir + '\\' + png_files[i]))
        labels.append(str(png_files[i][0]))
    
    # Return the training and validation images and labels in different variables
    training_images = normalized_images[:len(files)-n_validation]
    training_labels = labels[:len(files)-n_validation]
    validation_images = normalized_images[len(files)-n_validation:]
    validation_labels = labels[len(files)-n_validation:]

    return training_images, training_labels, validation_images, validation_labels


def generate_data_set(n_samples, data_dir):
    """
    Generate n_samples noisy images by using generate_noisy_image(), and store them in data_dir.

    Arguments
    ---------
    n_samples : int
        Number of train/test examples to generate
    data_dir : str in [TRAINING_IMAGE_DIR, TEST_IMAGE_DIR]
        Directory for storing images
    """

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)  # Generate a directory for data set storage, if not already present

    for i in range(n_samples):
        # Pick a random rank and convert it to a noisy image through generate_noisy_image().
        rank = random.choice(LABELS)
        img = generate_noisy_image(rank, random.uniform(0, 0.5))

        img.save(f"{data_dir}/{rank}_{i}.png")  # The filename encodes the original label for training/testing

def generate_noisy_image(rank, noise_level):
    """
    Generate a noisy image of the specified rank with the given noise_level. Furthermore, images will also be rotated.
    Maximum rotation is given by the ROTATE_MAX_ANGLE global variable. 

    Arguments
    ---------
    rank : str in ['J', 'Q', 'K']
        Original card rank.
    noise_level : int between zero and one
        Probability with which a given pixel is randomized.

    Returns
    -------
    noisy_img : Image
        A noisy image representation of the card rank.
    """

    if not 0 <= noise_level <= 1:
        raise ValueError(f"Invalid noise level: {noise_level}, value must be between zero and one")
    if rank not in LABELS:
        raise ValueError(f"Invalid card rank: {rank}")

    # Create rank image from text
    font = ImageFont.truetype(random.choice(FONTS), size = IMAGE_SIZE - 6)  # Pick a random font
    img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), color = 255)
    draw = ImageDraw.Draw(img)
    (text_width, text_height) = draw.textsize(rank, font = font)  # Extract text size
    draw.text(((IMAGE_SIZE - text_width) / 2, (IMAGE_SIZE - text_height) / 2 - 4), rank, fill = 0, font = font)

    # Random rotate transformation
    img = img.rotate(random.uniform(-ROTATE_MAX_ANGLE, ROTATE_MAX_ANGLE), expand = False, fillcolor = '#FFFFFF')
    pixels = list(img.getdata())  # Extract image pixels

    # Introduce random noise
    for (i, _) in enumerate(pixels):
        if random.random() <= noise_level:
            pixels[i] = random.randint(0, 255)  # Replace a chosen pixel with a random intensity

    # Save noisy image
    noisy_img = Image.new('L', img.size)
    noisy_img.putdata(pixels)

    return noisy_img

if __name__=='__main__':
    generate_data_set(1000, TEST_IMAGE_DIR)