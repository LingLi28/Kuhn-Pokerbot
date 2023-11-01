from data_sets import *
import shutil

TEST_DIR = os.path.dirname(os.path.abspath(__file__))  # Mark the test root directory
TRAINING_IMAGE_TEST_DIR = os.path.join(TEST_DIR, "data_sets", "training_images")
TEST_IMAGE_TEST_DIR = os.path.join(TEST_DIR, "data_sets", "test_images")
LABELS = ['J', 'Q', 'K']  # Possible card labels


class TestDataSets:
    # You don't need to test generate_data_set() (that will take too much time to run)
    def test_generate_data(self):
        # Clear test folder and generate one test picture
        shutil.rmtree(TRAINING_IMAGE_TEST_DIR)
        generate_data_set(1, TRAINING_IMAGE_TEST_DIR)
        files = os.listdir(TRAINING_IMAGE_TEST_DIR)

        # Check if indeed one file is created
        assert len(files) == 1

        # Check if indeed the first pixture has number 0
        assert files[0].endswith('0.png') == True

    def test_normalize_image(self):
        # Open image and use normalize function
        file = os.listdir(TRAINING_IMAGE_TEST_DIR)
        image = Image.open(TRAINING_IMAGE_TEST_DIR + '\\' + file[0])
        normalized_image = normalize_image(image)
        width, height = image.size
        pixels_image = list(image.getdata())

        # Check if the image still has the same dimensions
        assert len(normalized_image) == width
        assert len(normalized_image[0]) == height

        # Check if maximum value is 1 and minimum is 0 and check if there used to be higher values
        assert max(pixels_image) > 1.0
        assert max(sum(normalized_image, [])) == 1.0
        assert min(sum(normalized_image, [])) == 0

    def test_load_dataset(self):
        # Clear the test folder and create a new dataset of 10 pictures
        shutil.rmtree(TRAINING_IMAGE_TEST_DIR)
        generate_data_set(10, TRAINING_IMAGE_TEST_DIR)
        files = os.listdir(TRAINING_IMAGE_TEST_DIR)

        # Load the dataset and use 2 validation images
        training_images, training_labels, validation_images, validation_labels = load_data_set(TRAINING_IMAGE_TEST_DIR, 2)

        # Check if indeed 10 images are made
        assert len(files) == 10

        # Check if 2 images are used as validation 
        assert len(training_images) == 8
        assert len(training_labels) == 8
        assert len(validation_images) == 2
        assert len(validation_labels) == 2

        # Check if normalization is okay
        assert max(sum(training_images[0], [])) == 1.0
        assert min(sum(training_images[0], [])) == 0

        # Check if labels are strings, only contain one letter and are a possible labels
        assert type(training_labels[0]) == str
        assert len(training_labels[0]) == 1
        assert LABELS.count(training_labels[1]) == 1
        
