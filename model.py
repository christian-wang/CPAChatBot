
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
from chatbot_data import encoder_inp, decoder_final_output, decoder_inp, vocab
#13 is the length of input we are passing
encoder_input = Input(shape=(13,))
decoder_input = Input(shape=(13,))

VOCAB_SIZE = len(vocab)
embedding = Embedding(VOCAB_SIZE+1,output_dim=50, input_length=13, trainable = True )

encoder_embed = embedding(encoder_input)
encoder_lstm = LSTM(400,return_sequences=True, return_state=True)
encoder_op, h, c = encoder_lstm(encoder_embed)
encoder_states =[h,c]


decoder_embed = embedding(decoder_input)
decoder_lstm = LSTM(400,return_sequences=True, return_state=True)

decoder_op, _, _ = decoder_lstm(encoder_embed,initial_state=encoder_states)

dense = Dense(VOCAB_SIZE, activation='softmax')

dense_op = dense(decoder_op)

model = Model([encoder_input,decoder_input],dense_op)

model.compile(loss='categorical_crossentropy', metrics=['acc'],optimizer='adam')

model.fit([encoder_inp, decoder_inp], decoder_final_output, epochs=40)