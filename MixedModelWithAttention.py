from tensorflow.python.keras import Model
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Activation, Dot
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras import backend
from attention import Attention

backend.clear_session()
input_list = []
output_list = []
for _ in range(7):
    input_x = Input(shape=(24,))
    x = Dense(96, activation="relu")(input_x)
    x = Dense(48, activation="relu")(x)
    x = Dense(24, activation="relu")(x)
    #x = Model(inputs=input_x, outputs=x)
    input_y = Input(shape=(7,24))
    y = LSTM(24, activation = "tanh", return_sequences= False)(input_y)
    y = Activation('sigmoid')(y)
    #y = Attention(units=32)(y)
    #y = Model(inputs=input_y, outputs=y)
    xy = Dot(axes=(1,1))([y, x])
    #xy = concatenate([x.output, y.output])
    model = Model(inputs=[input_x, input_y], outputs=xy)
    input_list.append(input_x)
    input_list.append(input_y)
    output_list.append(model.output)

merge_m = concatenate([output_list[i] for i in range(7)])
z = Dense(96, activation="relu")(merge_m)
z = Dense(48, activation="relu")(z)
z = Dense(24, activation="relu")(z)

input_h = Input(shape=(7,24))
h = LSTM(24, activation = "tanh", return_sequences= True)(input_h)
h = Attention(units=32)(h)
h = Model(inputs=input_h, outputs=h)
input_list.append(input_h)
hz = concatenate([z, h.output])
k = Dense(96, activation="relu")(hz)
k = Dropout(0.3)(k)
k = Dense(48, activation="relu")(k)
k = Dropout(0.3)(k)
k = Dense(24, activation="relu")(k)
model_set = Model(inputs=[input_list[i] for i in range(15)], outputs=k)
model_set.compile(optimizer="adam",loss="mse")
model_set.save('MixedModelWithAttention.h5')