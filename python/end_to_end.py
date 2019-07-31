import h5py

from keras.models import Sequential
from keras.layers import Dense

from rlgames.agents.predict import DeepLearningAgent, load_prediction_agent
from rlgames.data_processor.parallel_processor import DataProcessor
from rlgames.encoders.base import get_encoder_by_name
from rlgames.httpfrontend import get_web_app
from rlgames.imitation_models.small import layers

board_sz = 19
nb_classes = board_sz * board_sz
encoder = get_encoder_by_name('oneplane', board_sz)
processor = DataProcessor('/home/armandli/rlgames/data', encoder=encoder.name())

X, y = processor.load_data(num_samples=100)

input_shape = (encoder.num_planes, board_sz, board_sz)
model = Sequential()
network_layers = layers(input_shape)
for layer in network_layers:
    model.add(layer)
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(X, y, batch_size=128, epochs=20, verbose=1)

agent_filename = '/home/armandli/rlgames/data/checkpoints/oneplane_large_imitation_agent.h5'
agent_file = h5py.File(agent_filename, 'w')
deep_learning_bot = DeepLearningAgent(model, encoder)
deep_learning_bot.serialize(agent_file)

model_file = h5py.File(agent_filename, "r")
bot_from_file = load_prediction_agent(model_file)

#TODO: this webapp is broken, cannot call the model to predict
web_app = get_web_app({'predict' : bot_from_file})
web_app.run()
