from data_sets import *
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten, Reshape, Dropout
from tensorflow.keras.callbacks import EarlyStopping

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Current file marks the root directory
MODEL_PATH = os.path.join(ROOT_DIR, "models")
CARD_LABELS = {"J": [1, 0, 0], "Q": [0, 1, 0], "K": [0, 0, 1]}

# Apply seed for reproducibility
SEED = 17102023
tf.random.set_seed(SEED)

# Apply seed for reproducibility
SEED = 17102023
tf.random.set_seed(SEED)

def build_model(filters, 
                kernel_size,
                units, 
                activation="relu", 
                ll_activation="linear",
                padding="same", 
                use_bias=True, 
                kernel_initializer="he_uniform",
                loss="mse",
                optimizer="adam"):
    """
    Prepare the model.
    
    Arguments
    ---------
    filters : int
        Number of filters used in the convolution layers.
    kernel_size : int
        Kernel size in the convolution layers.
    units : int
        Number of units in the dense layers.
    activation : str
        Activation function that is used in the hidden layers.
    ll_activation : str
        Activation function for the last/output layer.
    padding : str
        Padding type which is used in the convolution layer.
    use_bias : bool
        Specify if a bias is present.
    kernel_initializer : str
        Function that initializes the weight kernel.
    loss : str
        Loss function that is used durring training.
    optimizer : str
        Optimizer function that is used durring training.
    
    Returns
    -------
    model : model class from any toolbox you choose to use.
        Model definition (untrained).
    """
    model =  Sequential()
    
    # Input layer
    # Input layer
    model.add(Input(shape=(32, 32)))
    model.add(Reshape((32, 32, 1)))
    
    # Feature extraction
    for _ in range(2):
        model.add(Conv2D(filters, 
                        kernel_size, 
                        activation=activation, 
                        padding=padding, 
                        use_bias=use_bias, 
                        kernel_initializer = kernel_initializer,
                        kernel_regularizer = "L2"))
        model.add(Conv2D(filters, 
                        kernel_size, 
                        activation=activation, 
                        padding=padding, 
                        use_bias=use_bias, 
                        kernel_initializer = kernel_initializer))
        model.add(MaxPool2D())
        
    model.add(Conv2D(filters, 
                    3, 
                    activation=activation, 
                    padding=padding, 
                    use_bias=use_bias, 
                    kernel_initializer = kernel_initializer))
    
    model.add(MaxPool2D())
    
    # Categorization
    model.add(Flatten())
    for _ in range(2):
        model.add(Dense(units, 
                        activation=activation, 
                        use_bias=use_bias,
                        kernel_initializer = kernel_initializer))
        model.add(Dropout(0.3))
        model.add(Dense(units, 
                        activation=activation, 
                        use_bias=use_bias,
                        kernel_initializer = kernel_initializer,
                        kernel_regularizer = "L2"))
        
    model.add(Dense(3, activation=ll_activation))
    
    model.compile(loss=loss, optimizer=optimizer)
    
    # Save untrained model
    os.makedirs(MODEL_PATH, exist_ok=True)
    model_file = os.path.join(MODEL_PATH, "model.h5")
    model.save(model_file)
    return model

def train_model(model, 
                n_validation, 
                epochs=10, 
                batch_size=32, 
                patience=10,
                write_to_file=False):
    """
    Fit the model on the training data set.

    Arguments
    ---------
    model : model class
        Model structure to fit, as defined by build_model().
    n_validation : int
        Number of training examples used for cross-validation.
    epochs : int
        Number of epochs the model is trained on.
    batch_size : int
        Size of the batches after which the loss function is minimised.
    patience : int
        Number of epochs that is waited, if the models performance 
        does not increase, before stopping.
    write_to_file : bool
        Write model to file; can later be loaded through load_model().

    Returns
    -------
    model : model class
        The trained model.
    """
    # Load training and validation data  # Load training and validation data
    training_images, training_labels, validation_images, validation_labels = \
        load_data_set(TRAINING_IMAGE_DIR, n_validation)
        
    training_labels = encode(training_labels)
    validation_labels = encode(validation_labels)
    
    # Convert lists to tensors, in order to speed up initialization
    training_labels = tf.convert_to_tensor(training_labels, dtype=tf.float32)
    validation_labels = tf.convert_to_tensor(validation_labels, dtype=tf.float32)
    training_images = tf.convert_to_tensor(training_images, dtype=tf.float32)
    validation_images = tf.convert_to_tensor(validation_images, dtype=tf.float32)
    
    # Callbacks for training
    es_callback = EarlyStopping(monitor = "val_loss", patience = patience, restore_best_weights=True)
    keras_callbacks = [es_callback]
        
    hist = model.fit(training_images, 
                     training_labels, 
                     validation_data=(validation_images, validation_labels),
                     epochs=epochs, 
                     verbose=1, 
                     batch_size=batch_size, 
                     callbacks=keras_callbacks )
    
    # Save model
    # Save model
    if write_to_file:
        os.makedirs(MODEL_PATH, exist_ok=True)
        model_file = os.path.join(MODEL_PATH, "model.h5")
        model.save(model_file)
    
    return model


def load_model():
    """
    Load a model from file.

    Returns
    -------
    model : model class
        Previously trained model.
    """
    model_file = os.path.join(MODEL_PATH, "model.h5")
    model = tf.keras.models.load_model(model_file)
    return model


def evaluate_model(model):
    """
    Evaluate model on the test set.

    Arguments
    ---------
    model : model class
        Trained model.

    Returns
    -------
    score : float
        A measure of model performance.
    """
    test_images, test_labels, _, _ = load_data_set(TEST_IMAGE_DIR)
    test_labels = encode(test_labels)
    
    # Convert to list into tensor, in order to decrease computation time
    test_labels = tf.convert_to_tensor(test_labels, dtype=tf.float32)
    test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)
    
    score =  model.evaluate(test_images, test_labels)
    return score    


def identify(raw_image, model):
    """
    Use model to classify a single card image.

    Arguments
    ---------
    raw_image : Image
        Raw image to classify.
    model : model class
        Trained model.

    Returns
    -------
    rank : str in ['J', 'Q', 'K']
        Estimated card rank.
    """
    image = normalize_image(raw_image)
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    
    # Change softmax normalization into one and zeros
    # Change softmax normalization into one and zeros
    cast = np.zeros_like(pred)
    cast[:, np.argmax(pred)] = 1

    rank = decode(cast.tolist())
    return rank

def encode(train_labels):
    """
    One-hot encode labels.

    Arguments
    ---------
    train_labels : list
        A list containing the labels for training in ['J', 'Q', 'K']

    Returns
    -------
    train_labels_enc : list
        One-hot encoded labels
    """
    """
    One-hot encode labels.

    Arguments
    ---------
    train_labels : list
        A list containing the labels for training in ['J', 'Q', 'K']

    Returns
    -------
    train_labels_enc : list
        One-hot encoded labels
    """
    train_labels_enc = [CARD_LABELS[i] for i in train_labels]
    return train_labels_enc

def decode(train_labels_enc):
    """
    Return labels from one-hot encoded input.

    Arguments
    ---------
    train_labels_enc : list
        A list of one-hot encoded labels.

    Returns
    -------
    train_labels : list
        A list containing a str in ['J', 'Q', 'K']
    """
    """
    Return labels from one-hot encoded input.

    Arguments
    ---------
    train_labels_enc : list
        A list of one-hot encoded labels.

    Returns
    -------
    train_labels : list
        A list containing a str in ['J', 'Q', 'K']
    """
    train_label = [[key for key, value in CARD_LABELS.items() if i==value] for i in train_labels_enc]
    return sum(train_label, []) # reduce dimensions of list
    return sum(train_label, []) # reduce dimensions of list