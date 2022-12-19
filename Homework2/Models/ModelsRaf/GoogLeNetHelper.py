def build_GoogLeNet_model(inputShape, nClasses, learningRate=0.001, momentum=0.9, loss="categorical_crossentropy",
                          metrics=["accuracy"]):
    """
    Builds a GoogLeNet model for data series with Conv1D.
    """
    # Define the model input
    input = Input(shape=inputShape)

    # Apply convolutional layers
    x = Conv1D(filters=32, kernel_size=3, strides=1, padding="same")(input)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    # Apply Inception modules
    x = Inception1D(filters=128, strides=1, padding="same")(x)
    x = Inception1D(filters=128, strides=1, padding="same")(x)

    # Apply global pooling
    x = GlobalAveragePooling1D()(x)

    # Add a fully connected layer
    x = Dense(units=nClasses)(x)
    x = Activation("softmax")(x)

    # Define the model
    model = Model(inputs=input, outputs=x)

    # Compile the model
    model.compile(optimizer=SGD(learning_rate=learningRate, momentum=momentum), loss=loss, metrics=metrics)

    return model


def Inception1D(filters, strides=1, padding="same"):
    """
    Builds an Inception module for 1D data with Conv1D.
    """

    def inception(x):
        # Define the branch structures
        branch1x1 = Conv1D(filters=filters, kernel_size=1, strides=strides, padding=padding)(x)
        branch1x1 = Activation("relu")(branch1x1)
        branch1x1 = BatchNormalization()(branch1x1)

        branch3x3 = Conv1D(filters=filters, kernel_size=1, strides=strides, padding=padding)(x)
        branch3x3 = Activation("relu")(branch3x3)
        branch3x3 = BatchNormalization()(branch3x3)
        branch3x3 = Conv1D(filters=filters, kernel_size=3, strides=strides, padding=padding)(branch3x3)
        branch3x3 = Activation("relu")(branch3x3)
        branch3x3 = BatchNormalization()(branch3x3)

        branch5x5 = Conv1D(filters=filters, kernel_size=1, strides=strides, padding=padding)(x)
        branch5x5 = Activation("relu")(branch5x5)
        branch5x5 = BatchNormalization()(branch5x5)
        branch5x5 = Conv1D(filters=filters, kernel_size=5, strides=strides, padding=padding)(branch5x5)
        branch5x5 = Activation("relu")(branch5x5)
        branch5x5 = BatchNormalization()(branch5x5)

        branchpool = MaxPooling1D(pool_size=3, strides=strides, padding=padding)(x)
        branchpool = Conv1D(filters=filters, kernel_size=1, strides=strides, padding=padding)(branchpool)
        branchpool = Activation("relu")(branchpool)
        branchpool = BatchNormalization()(branchpool)

        # Concatenate the branches
        x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=2)

        return x

    return inception


# build a network to classify data series with Conv1D
def build_4_layers_conv1d_network(input_shape, classes, learning_rate=0.001, momentum=0.9,
                                  loss="categorical_crossentropy",
                                  metrics=["accuracy"]):
    """
    Builds 4 layers Conv1D network for data series.
    """
    # Define the model input
    input = Input(shape=input_shape)

    # Apply convolutional layers
    x = Conv1D(filters=32, kernel_size=7, strides=1, padding="same")(input)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=64, kernel_size=5, strides=1, padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # apply convolutional layers with sigmoid activation
    x = Conv1D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x = Activation("sigmoid")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Apply a dense layer
    x = Flatten()(x)
    x = Dense(units=classes)(x)

    # Apply global pooling
    x = GlobalAveragePooling1D()(x)

    # Add a fully connected layer
    x = Dense(units=classes)(x)
    x = Activation("softmax")(x)

    # Define the model
    model = Model(inputs=input, outputs=x)

    # Compile the model
    model.compile(optimizer=SGD(learning_rate=learning_rate, momentum=momentum), loss=loss, metrics=metrics)

    return model

