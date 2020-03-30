from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, GaussianNoise, InputLayer


def get_model(noise=0.1, nn_scale=8, lookback=4):
    sample_shape = (30, 64, 2*lookback) # lat, lon, channel
    model = Sequential()
    model.add(InputLayer(input_shape=sample_shape))
    if noise:
        model.add(GaussianNoise(stddev=noise))
    model.add(Conv2D(64*nn_scale, kernel_size=(4, 8), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(32*nn_scale, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(16*nn_scale, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2D(8*nn_scale, kernel_size=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(8*nn_scale))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer='adam', loss='huber_loss')
    return model
