# function to build the GoogLeNet model for data series with Conv1D
def buildGoogLeNetModel_Copilot(inputShape, nClasses, learningRate=0.001, momentum=0.9, loss="categorical_crossentropy",
                        metrics=["accuracy"]):
    """
    Builds a GoogLeNet model for data series with Conv1D.
    """
    inputLayer = Input(shape=inputShape)
    conv1 = Conv1D(filters=64, kernel_size=7, strides=2, padding="same", activation="relu")(inputLayer)
    pool1 = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv1)
    conv2 = Conv1D(filters=64, kernel_size=1, strides=1, padding="same", activation="relu")(pool1)
    conv3 = Conv1D(filters=192, kernel_size=3, strides=1, padding="same", activation="relu")(conv2)
    pool2 = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv3)
    inception3a = inceptionModule(inputLayer=pool2, nFilters=64, nFilters1x1=96, nFilters3x3=128, nFilters5x5=16,
                                  nFiltersPoolProj=32)
    inception3b = inceptionModule(inputLayer=inception3a, nFilters=128, nFilters1x1=128, nFilters3x3=192,
                                  nFilters5x5=32, nFiltersPoolProj=96)
    pool3 = MaxPooling1D(pool_size=3, strides=2, padding="same")(inception3b)
    inception4a = inceptionModule(inputLayer=pool3, nFilters=192, nFilters1x1=96, nFilters3x3=208, nFilters5x5=16,
                                  nFiltersPoolProj=48)
    inception4b = inceptionModule(inputLayer=inception4a, nFilters=160, nFilters1x1=112, nFilters3x3=224,
                                  nFilters5x5=24, nFiltersPoolProj=64)
    inception4c = inceptionModule(inputLayer=inception4b, nFilters=128, nFilters1x1=128, nFilters3x3=256,
                                  nFilters5x5=24, nFiltersPoolProj=64)
    inception4d = inceptionModule(inputLayer=inception4c, nFilters=112, nFilters1x1=144, nFilters3x3=288,
                                  nFilters5x5=32, nFiltersPoolProj=64)
    inception4e = inceptionModule(inputLayer=inception4d, nFilters=256, nFilters1x1=160, nFilters3x3=320,
                                  nFilters5x5=32, nFiltersPoolProj=128)
    pool4 = MaxPooling1D(pool_size=3, strides=2, padding="same")(inception4e)
    inception5a = inceptionModule(inputLayer=pool4, nFilters=256, nFilters1x1=160, nFilters3x3=320, nFilters5x5=32,
                                  nFiltersPoolProj=128)
    inception5b = inceptionModule(inputLayer=inception5a, nFilters=384, nFilters1x1=192, nFilters3x3=384,
                                  nFilters5x5=48, nFiltersPoolProj=128)
    pool5 = AveragePooling1D(pool_size=7, strides=1)(inception5b)
    dropout = Dropout(rate=0.4)(pool5)
    fc1 = Flatten()(dropout)
    fc2 = Dense(units=nClasses, activation="softmax")(fc1)
    model = Model(inputs=inputLayer, outputs=fc2)
    model.compile(optimizer=SGD(lr=learningRate, momentum=momentum), loss=loss, metrics=metrics)
    return model


# inception module of the buildGoogLeNetModel_Copilot function
def inceptionModule(inputLayer, nFilters, nFilters1x1, nFilters3x3, nFilters5x5, nFiltersPoolProj):
    """
    Inception module of the buildGoogLeNetModel_Copilot function.
    """
    conv1x1 = Conv1D(filters=nFilters1x1, kernel_size=1, strides=1, padding="same", activation="relu")(inputLayer)
    conv3x3 = Conv1D(filters=nFilters3x3, kernel_size=3, strides=1, padding="same", activation="relu")(inputLayer)
    conv5x5 = Conv1D(filters=nFilters5x5, kernel_size=5, strides=1, padding="same", activation="relu")(inputLayer)
    pool = MaxPooling1D(pool_size=3, strides=1, padding="same")(inputLayer)
    poolProj = Conv1D(filters=nFiltersPoolProj, kernel_size=1, strides=1, padding="same", activation="relu")(pool)
    outputLayer = concatenate([conv1x1, conv3x3, conv5x5, poolProj], axis=-1)
    return outputLayer




