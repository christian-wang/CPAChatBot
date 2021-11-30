import torch
import torch.nn as nn
import random
import os
import codecs

from torch import optim

from data_prep4 import get_question_answers, trimRareWords, batch2TrainData, load_dialogues, load_exchanges
from hyperparams import learning_rate, decoder_learning_ratio, save_every, attn_model, hidden_size, encoder_n_layers, decoder_n_layers, dropout, small_batch_size
from model import EncoderRNN, LuongAttnDecoderRNN, CPAChatBot

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus = "corpus"


delimiter = '\t'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

conversations = []
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

print("\nProcessing corpus...")
dialogues = load_dialogues(os.path.join(corpus, "movie_lines.txt"))
print("\nLoading conversations...")
exchanges = load_exchanges(os.path.join(corpus, "movie_conversations.txt"))

save_dir = os.path.join("data", "save")
question_answers, vocab = get_question_answers(dialogues, exchanges)
pairs = trimRareWords(vocab, question_answers)

batches = batch2TrainData(vocab, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)

loadFilename = None

checkpoint = None

if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    vocab.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
embedding = nn.Embedding(len(vocab), hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, len(vocab), decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

encoder.train()
decoder.train()

print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
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
cpa.trainIters(save_dir, save_every, loadFilename, checkpoint)

encoder.eval()
decoder.eval()

# cpa.run()
