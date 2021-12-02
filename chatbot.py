import os.path

import torch
import torch.nn as nn
import argparse
import hyperparams as hp
from torch import optim
from preprocessing import get_question_answers, load_dialogues, load_exchanges
from model import EncoderRNN, DecoderRNN, CPAChatBot


def parse_args():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--save', type=str, default=None, help='Model save path')
    parser.add_argument('--load', type=str, default=None, help='Model load path')

    prep_parser = parser.add_argument_group('Preprocessing')
    prep_parser.add_argument('--conversations_file', type=str, default='corpus/movie_conversations.txt',
                             help='Path to movie_conversations.txt')
    prep_parser.add_argument('--lines_file', type=str, default='corpus/movie_lines.txt', help='Path to movie_lines.txt')
    prep_parser.add_argument('--max-sentence-length', type=int, default=hp.MAX_SENTENCE_LENGTH,
                             help='Maximum number of words in question')
    prep_parser.add_argument('--min-word-count', type=int, default=hp.MIN_WORD_COUNT,
                             help='Minimum number of times a word can appear to be included in vocabulary')

    train_parser = parser.add_argument_group('Training arguments and hyperparameters')
    train_parser.add_argument('--iterations', type=int, default=hp.ITERATIONS, help='Number of iterations to train')
    train_parser.add_argument('--encoder-learning-rate', type=float, default=hp.ENCODER_LEARNING_RATE,
                              help='Learning rate for encoder')
    train_parser.add_argument('--decoder-learning-rate', type=float, default=hp.DECODER_LEARNING_RATE,
                              help='Learning rate for decoder')
    train_parser.add_argument('--save-every', type=int, default=hp.SAVE_EVERY, help='Save model every n iterations')
    train_parser.add_argument('--encoder-layers', type=int, default=hp.ENCODER_LAYERS, help='Number of encoder layers')
    train_parser.add_argument('--decoder-layers', type=int, default=hp.DECODER_LAYERS, help='Number of decoder layers')
    train_parser.add_argument('--dropout', type=float, default=hp.DROPOUT, help='Dropout rate')
    train_parser.add_argument('--batch-size', type=int, default=hp.BATCH_SIZE, help='Batch size')
    train_parser.add_argument('--attention-type', type=str, choices=('dot', 'mul', 'add'), default=hp.ATTENTION_TYPE,
                              help='Attention type')

    args = parser.parse_args()

    # adjust hyperparameters in module so we don't need to pass them as arguments
    hp.MAX_SENTENCE_LENGTH = args.max_sentence_length
    hp.MIN_WORD_COUNT = args.min_word_count
    hp.ITERATIONS = args.iterations
    hp.ENCODER_LEARNING_RATE = args.encoder_learning_rate
    hp.DECODER_LEARNING_RATE = args.decoder_learning_rate
    hp.SAVE_EVERY = args.save_every
    hp.ENCODER_LAYERS = args.encoder_layers
    hp.DECODER_LAYERS = args.decoder_layers
    hp.DROPOUT = args.dropout
    hp.BATCH_SIZE = args.batch_size
    hp.ATTENTION_TYPE = args.attention_type

    return args


def main(in_notebook=False):
    """
    Main function
    :param in_notebook: whether or not we are running in a notebook
    """
    if in_notebook:
        # run in notebook mode, so we don't parse command line arguments
        load_file = 'model.pt' if os.path.isfile('model.pt') else None
        save_file = 'model.pt'
        lines_file = 'corpus/movie_lines.txt'
        conversations_file = 'corpus/movie_conversations.txt'
    else:
        # run in command-line mode
        args = parse_args()
        load_file = args.load
        save_file = args.save
        lines_file = args.lines_file
        conversations_file = args.conversations_file

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the data
    print("\nReading movie lines...")
    dialogues = load_dialogues(lines_file)

    # load the conversations
    print("Loading conversations...")
    exchanges = load_exchanges(conversations_file)

    # build the vocabulary
    print("Building vocabulary...")
    question_answers, vocab = get_question_answers(dialogues, exchanges)

    # define variables so IDE doesn't worry
    embedding_sd = encoder_sd = decoder_sd = encoder_optimizer_sd = \
        decoder_optimizer_sd = prev_iteration = None

    if load_file:
        # load the model from file
        print("Loading model from file...")
        checkpoint = torch.load(load_file)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        vocab.__dict__ = checkpoint['voc_dict']
        prev_iteration = checkpoint['iteration']

    # create word embeddings
    embeddings = nn.Embedding(len(vocab), hp.HIDDEN_LAYER_DIM)

    if load_file:
        # load embeddings from file
        embeddings.load_state_dict(embedding_sd)

    print("Building neural nets...")
    encoder = EncoderRNN(embeddings)
    decoder = DecoderRNN(embeddings, len(vocab))
    if load_file:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.train()
    decoder.train()

    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=hp.ENCODER_LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=hp.DECODER_LEARNING_RATE)

    if load_file:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    for optimizer in (encoder_optimizer, decoder_optimizer):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    print("Training...")
    cpa = CPAChatBot(encoder, decoder, encoder_optimizer, decoder_optimizer,
                     embeddings, vocab, question_answers, device)

    if load_file:
        cpa.train(save_file, prev_iteration)
    else:
        cpa.train(save_file)

    encoder.eval()
    decoder.eval()

    cpa.run()


if __name__ == "__main__":
    main(in_notebook=False)
