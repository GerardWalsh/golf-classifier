# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential


def build_model(base, classes, base_trainable=False):
	# Set the status of the base model (transfer learning)
	base.trainable = base_trainable

	model = Sequential()

	model.add(base)
	model.add(Flatten())
	model.add(Dense(16, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(classes, activation='softmax'))

	return model
