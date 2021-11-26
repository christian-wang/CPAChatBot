import argparse
import numpy as np
import pickle
import resource
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
from keras.preprocessing.sequence import pad_sequences

from chatbot_data import load_conversations, load_dialogues, get_question_answers, get_word_counts, create_vocab, \
    create_inputs, get_decoder_final_output, clean_text


# set maximum memory usage so process doesn't crash my machine
resource.setrlimit(resource.RLIMIT_AS, (9663676416, 9663676416))


class CPAChatBot:
    vocab = None
    inv_vocab = None

    encoder_inp = None
    decoder_inp = None
    encoder_input = None
    decoder_input = None
    encoder_states = None
    decoder_final_output = None

    decoder_lstm = None
    decoder_embed = None
    dense = None

    def __init__(self, max_question_length, max_conversations, min_word_count):
        self.max_question_length = max_question_length
        self.max_conversations = max_conversations
        self.min_word_count = min_word_count

    def prepare_data(self, conversations_file, lines_file):
        conversations = load_conversations(conversations_file)
        lines = load_dialogues(lines_file)
        question_answers = get_question_answers(conversations, lines)
        word2count = get_word_counts(question_answers)
        self.vocab, self.inv_vocab = create_vocab(word2count, self.min_word_count)
        self.encoder_inp, self.decoder_inp = create_inputs(question_answers, self.vocab)
        self.decoder_final_output = get_decoder_final_output(self.decoder_inp, self.vocab)

    def train(self, epochs):
        # 13 is the length of input we are passing
        self.encoder_input = Input(shape=(self.max_question_length,))
        self.decoder_input = Input(shape=(self.max_question_length,))

        vocab_size = len(self.vocab)
        embedding = Embedding(vocab_size + 1, output_dim=50, input_length=self.max_question_length, trainable=True)

        encoder_embed = embedding(self.encoder_input)
        encoder_lstm = LSTM(400, return_sequences=True, return_state=True)
        encoder_op, h, c = encoder_lstm(encoder_embed)
        self.encoder_states = [h, c]

        self.decoder_embed = embedding(self.decoder_input)
        self.decoder_lstm = LSTM(400, return_sequences=True, return_state=True)

        decoder_op, _, _ = self.decoder_lstm(self.decoder_embed, initial_state=self.encoder_states)

        self.dense = Dense(vocab_size, activation='softmax')

        dense_op = self.dense(decoder_op)

        model = Model([self.encoder_input, self.decoder_input], dense_op)

        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

        model.fit([self.encoder_inp, self.decoder_inp], self.decoder_final_output, epochs=epochs)
        del self.encoder_inp, self.decoder_inp, self.decoder_final_output

    def test(self):
        encoder_model = Model([self.encoder_input], self.encoder_states)

        # decoder Model
        decoder_state_input_h = Input(shape=(400,))
        decoder_state_input_c = Input(shape=(400,))

        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = self.decoder_lstm(self.decoder_embed, initial_state=decoder_states_inputs)

        decoder_states = [state_h, state_c]

        decoder_model = Model([self.decoder_input] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        print("##########################################")
        print("#       start chatting ver. 1.0          #")
        print("##########################################")

        prepro1 = ""
        while prepro1 != 'q':
            prepro1 = clean_text(input("you: "))

            prepro = [prepro1]

            txt = []
            for x in prepro:
                # x = "hello"
                lst = []
                for y in x.split():
                    # y = "hello"
                    try:
                        lst.append(self.vocab[y])
                        # vocab['hello'] = 454
                    except KeyError:
                        lst.append(self.vocab['<OUT>'])
                txt.append(lst)

            # txt = [[454]]
            txt = pad_sequences(txt, self.max_question_length, padding='post')

            # txt = [[454,0,0,0,.........13]]

            stat = encoder_model.predict(txt)

            empty_target_seq = np.zeros((1, 1))
            #   empty_target_seq = [0]

            empty_target_seq[0, 0] = self.vocab['<SOS>']
            #    empty_target_seq = [255]

            stop_condition = False
            decoded_translation = ''

            while not stop_condition:
                decoder_outputs, h, c = decoder_model.predict([empty_target_seq] + stat)

                decoder_concat_input = self.dense(decoder_outputs)
                # decoder_concat_input = [0.1, 0.2, .4, .0, ...............]

                sampled_word_index = np.argmax(decoder_concat_input[0, -1, :])
                # sampled_word_index = [2]

                sampled_word = self.inv_vocab[sampled_word_index] + ' '

                # inv_vocab[2] = 'hi'
                # sampled_word = 'hi '

                if sampled_word != '<EOS> ':
                    decoded_translation += sampled_word

                if sampled_word == '<EOS> ' or len(decoded_translation.split()) > self.max_question_length:
                    stop_condition = True

                empty_target_seq = np.zeros((1, 1))
                empty_target_seq[0, 0] = sampled_word_index
                # <SOS> - > hi
                # hi --> <EOS>
                stat = [h, c]

            print("chatbot attention : ", decoded_translation)
            print("==============================================")


def parse_args():
    parser = argparse.ArgumentParser()

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--save', type=str, default=None, help='Model save path')
    model_group.add_argument('--load', type=str, default=None, help='Model load path')

    parser.add_argument('--conversations_file', type=str, default='corpus/movie_conversations.txt',
                        help='Path to movie_conversations.txt')
    parser.add_argument('--lines_file', type=str, default='corpus/movie_lines.txt', help='Path to movie_lines.txt')
    parser.add_argument('--max-question-length', type=int, default=13, help='Maximum number of words in question')
    parser.add_argument('--max-conversations', type=int, default=30000,
                        help='Maximum number of conversations to train on')
    parser.add_argument('--min-word-count', type=int, default=5,
                        help='Minimum number of times a word can appear to be included in vocabulary')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs to train for')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.load:
        # read CPA model from file instead of training from scatch
        print(f'Loading CPA model from {args.load}')
        with open(args.load, 'rb') as f:
            cpa = pickle.load(f)
    else:
        # create new CPA model and train it
        conversations_file = args.conversations_file
        lines_file = args.lines_file

        max_question_length = args.max_question_length
        max_conversations = args.max_conversations
        min_word_count = args.min_word_count

        cpa = CPAChatBot(max_question_length, max_conversations, min_word_count)

        print("Preparing data...")
        cpa.prepare_data(conversations_file, lines_file)

        print("Training...")
        cpa.train(args.epochs)
        if args.save:
            print(f'Saving model to {args.save}')
            with open(args.save, 'wb') as f:
                pickle.dump(cpa, f)

    cpa.test()


if __name__ == '__main__':
    main()
