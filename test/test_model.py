import os
import pytest
import numpy as np
from model import build_model, encode, decode

TEST_DIR = os.path.dirname(os.path.abspath(__file__))  # Mark the test root directory
TRAINING_IMAGE_TEST_DIR = os.path.join(TEST_DIR, "data_sets", "training_images")
TEST_IMAGE_TEST_DIR = os.path.join(TEST_DIR, "data_sets", "test_images")


class TestModel:
    def test_build_model(self):
        model = build_model(1,1,1)
        # Test if model is trainable and has correct output shape
        assert model.trainable is True
        assert model.output_shape == (None, 3)
        
        # Test for correct input types
        with pytest.raises(TypeError):
            build_model("a", 1, 1)
            build_model(1, "a", 1)
            build_model(1, 1, "a")
            build_model(1,1,1, activation=0)
            build_model(1,1,1, ll_activation=0)
            build_model(1,1,1, use_bias="a")
        
        # Test for correct deserialization 
        with pytest.raises(ValueError):
            build_model(1,1,1, activation="random_string")
            build_model(1,1,1, ll_activation="random_string")
            build_model(1,1,1, padding="random_string")
            build_model(1,1,1, kernel_initializer="random_string")
            build_model(1,1,1, loss="random_string")
            build_model(1,1,1, optimizer="random_string")            
        
    def test_encode(self):
        # Test for correct encoding 
        assert np.array_equal(encode(["J"]), [[1, 0, 0]])
        assert np.array_equal(encode(["Q"]), [[0, 1, 0]])
        assert np.array_equal(encode(["K"]), [[0, 0, 1]])
        
    def test_decode(self):
        # Test for correct decoding
        assert np.array_equal(decode([[1, 0, 0]]), ["J"])
        assert np.array_equal(decode([[0, 1, 0]]), ["Q"])
        assert np.array_equal(decode([[0, 0, 1]]), ["K"])
