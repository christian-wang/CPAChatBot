import torch
import torch.nn as nn
import argparse
from torch import optim
from data_prep4 import get_question_answers, load_dialogues, load_exchanges
import hyperparams as hp
from model import EncoderRNN, LuongAttnDecoderRNN, CPAChatBot


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
    prep_parser.add_argument('--max-sentence-length', type=int, default=hp.MAX_SENTENCE_LENGTH, help='Maximum number of words in question')
    prep_parser.add_argument('--min-word-count', type=int, default=hp.MIN_WORD_COUNT,
                             help='Minimum number of times a word can appear to be included in vocabulary')

    train_parser = parser.add_argument_group('Training arguments and hyperparameters')
    train_parser.add_argument('--iterations', type=int, default=hp.ITERATIONS, help='Number of iterations to train')
    train_parser.add_argument('--learning-rate', type=float, default=hp.LEARNING_RATE, help='Learning rate')
    train_parser.add_argument('--save-every', type=int, default=hp.SAVE_EVERY, help='Save model every n iterations')
    train_parser.add_argument('--encoder-layers', type=int, default=hp.ENCODER_LAYERS, help='Number of encoder layers')
    train_parser.add_argument('--decoder-layers', type=int, default=hp.DECODER_LAYERS, help='Number of decoder layers')
    train_parser.add_argument('--dropout', type=float, default=hp.DROPOUT, help='Dropout rate')
    train_parser.add_argument('--batch-size', type=int, default=hp.BATCH_SIZE, help='Batch size')

    args = parser.parse_args()

    # adjust hyperparameters in module so we don't need to pass them as arguments
    hp.MAX_SENTENCE_LENGTH = args.max_sentence_length
    hp.MIN_WORD_COUNT = args.min_word_count
    hp.ITERATIONS = args.iterations
    hp.LEARNING_RATE = args.learning_rate
    hp.SAVE_EVERY = args.save_every
    hp.ENCODER_LAYERS = args.encoder_layers
    hp.DECODER_LAYERS = args.decoder_layers
    hp.DROPOUT = args.dropout
    hp.BATCH_SIZE = args.batch_size

    return args


def main(in_notebook=False):

    if in_notebook:
        load_file = None
        save_file = None
        lines_file = 'drive/MyDrive/chatbot2/movie_lines.txt'
        conversations_file = 'drive/MyDrive/chatbot2/movie_conversations.txt'
    else:
        args = parse_args()
        load_file = args.load
        save_file = args.save
        lines_file = args.lines_file
        conversations_file = args.conversations_file

    # uncomment these lines when running on command line

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nProcessing corpus...")
    dialogues = load_dialogues(lines_file)
    print("\nLoading conversations...")
    exchanges = load_exchanges(conversations_file)

    question_answers, vocab = get_question_answers(dialogues, exchanges)

    if load_file:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(load_file)
        # If loading a model trained on GPU to CPU
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        vocab.__dict__ = checkpoint['voc_dict']
        prev_iteration = checkpoint['iteration']

    print('Building encoder and decoder ...')
    embedding = nn.Embedding(len(vocab), hp.HIDDEN_LAYER_DIM)
    if load_file:
        embedding.load_state_dict(embedding_sd)
    encoder = EncoderRNN(hp.HIDDEN_LAYER_DIM, embedding, hp.ENCODER_LAYERS, hp.DROPOUT)
    decoder = LuongAttnDecoderRNN(hp.ATTENTION_TYPE, embedding, hp.HIDDEN_LAYER_DIM, len(vocab), hp.DECODER_LAYERS, hp.DROPOUT)
    if load_file:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    encoder.train()
    decoder.train()

    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=hp.LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=hp.LEARNING_RATE * hp.DECODER_LEARNING_RATIO)
    if load_file:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    print("Starting Training!")

    cpa = CPAChatBot(encoder, decoder, encoder_optimizer, decoder_optimizer,
                     embedding, vocab, question_answers, device)

    if load_file:
        cpa.trainIters(save_file, prev_iteration)
    else:
        cpa.trainIters(save_file)

    encoder.eval()
    decoder.eval()

    # cpa.run()


main(in_notebook=True)
