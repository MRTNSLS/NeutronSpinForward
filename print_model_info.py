from tensorflow.keras.models import load_model

ver = 30
model = load_model(f'models/my_model_{ver}.h5')
model.summary()

ver = 101
model = load_model(f'models/my_model_{ver}.h5')
model.summary()