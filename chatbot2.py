import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import re
from torchtext.legacy.data import Field, Example, Dataset, BucketIterator
from typing import Dict, List, Tuple, Union
import spacy
import numpy as np
import random
import math
import time

import model

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_en = spacy.load('en_core_web_sm')


class HyperParams:
    encoder_emb_dim = 256
    decoder_emb_dim = 256
    hidden_dim = 512
    enc_dropout = 0.5
    dec_dropout = 0.5
    batch_size = 1
    clip = 1


def tokenize(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text.strip() for tok in spacy_en.tokenizer(text) if tok.text.strip()]


QFIELD = Field(tokenize=tokenize,
               init_token='<sos>',
               eos_token='<eos>',
               lower=True)

AFIELD = Field(tokenize=tokenize,
               init_token='<sos>',
               eos_token='<eos>',
               lower=True)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


def clean_text(txt):
    # cleaning text, change uppercase to lowercase, remove punctuation.
    # Uses regular expressions to do so
    txt = txt.lower()
    txt = txt.replace("i'm", "i am")
    txt = txt.replace("he's", "he is")
    txt = txt.replace("she's", "she is")
    txt = txt.replace("that's", "that is")
    txt = txt.replace("what's", "what is")
    txt = txt.replace("where's", "where is")
    txt = txt.replace("'ll", " will")
    txt = txt.replace("'ve", " have")
    txt = txt.replace("'re", " are")
    txt = txt.replace("'d", " would")
    txt = txt.replace("won't", "will not")
    txt = txt.replace("can't", "cannot")
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt


def load_exchanges(file_name: str) -> List[List[str]]:
    """
    Return list of conversations from file_name

    :param file_name: path to file
    :return: List of [questionLineID, answerLineID] 2-lists
    """
    exchanges = []
    try:
        with open(file_name, encoding='utf-8', errors='ignore') as f:
            for line in f:
                conversation = line.strip().split(' +++$+++ ')[-1][1:-1].replace("'", "").split(", ")
                for i in range(len(conversation) - 1):
                    question_id = conversation[i]
                    answer_id = conversation[i + 1]
                    exchanges.append([question_id, answer_id])

    except FileNotFoundError:
        print(f"Conversations file not found at \"{file_name}\"")
        exit(1)

    return exchanges


def load_dialogues(file_name: str) -> Dict[str, str]:
    dialogues = dict()
    try:
        with open(file_name, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_id, character_id, movie_id, character_name, text = line.rstrip('\n').split(' +++$+++ ')
                dialogue = clean_text(text)
                dialogues[line_id] = dialogue
    except FileNotFoundError:
        print(f"Lines file not found at \"{file_name}\"")
        exit(1)
    return dialogues


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)  # no cell state!

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # context = [n layers * n directions, batch size, hid dim]

        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        emb_con = torch.cat((embedded, context), dim=2)

        # emb_con = [1, batch size, emb dim + hid dim]

        output, hidden = self.rnn(emb_con, hidden)

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # seq len, n layers and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)),
                           dim=1)

        # output = [batch size, emb dim + hid dim * 2]

        prediction = self.fc_out(output)

        # prediction = [batch size, output dim]

        return prediction, hidden


def decoder_out_to_text(decoder_out):
    words = []
    for word_id in decoder_out.topk(1)[1][:, 0].data:
        word = AFIELD.vocab.itos[word_id]
        if word == AFIELD.eos_token:
            break
        words.append(word)
    return " ".join(words)


def text_to_encoder_input(text):
    tokens = [QFIELD.init_token] + tokenize(clean_text(text)) + [QFIELD.eos_token]
    return torch.tensor([QFIELD.vocab.stoi[t] for t in tokens])


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is the context
        context = self.encoder(src)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs

    def get_response(self, user_input):
        src = text_to_encoder_input(user_input).unsqueeze(1)
        trg = torch.zeros(13, 1, dtype=torch.long).to(self.device)
        trg[0][0] = AFIELD.vocab.stoi[AFIELD.init_token]
        output = self(src, trg, 0)
        output_text = decoder_out_to_text(output)
        return output_text


def create_datasets(conversations_file, lines_file, split_ratios: Union[List, float] = 0.8):
    exchanges = load_exchanges(conversations_file)
    lines = load_dialogues(lines_file)
    examples = []

    fields = [('src', QFIELD), ('trg', AFIELD)]

    for question_id, answer_id in exchanges:
        question = lines[question_id]
        answer = lines[answer_id]
        if len(question) > 13:
            continue
        answer = ' '.join(answer.split()[:11])
        # creates an example instance from the question ID and answer ID
        examples.append(Example.fromlist(
            [question, answer],
            fields))

    dataset = Dataset(examples, fields)

    split_datasets = dataset.split(split_ratios)

    # split dataset into training, validation, and testing
    # lengths = [int(len(dataset) * ratios[0]), len(dataset) - int(len(dataset) * ratios[0])]

    # train_data, test_data = random_split(dataset, lengths,
    #                                      generator=torch.Generator().manual_seed(SEED))
    return split_datasets


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
    parser.add_argument('--min-word-count', type=int, default=2,
                        help='Minimum number of times a word can appear to be included in vocabulary')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs to train for')

    args = parser.parse_args()
    return args


def train_epoch(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):

        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def test(model):
    model.eval()

    with torch.no_grad():
        # while True:
        #     user_input = input("You: ")
        #     print("CPA:", model.get_response(user_input))
        model.get_response("hello my name is bob")


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_model(model, train_iterator, optimizer, criterion, epochs):

    for epoch in range(epochs):

        start_time = time.time()

        train_loss = train_epoch(model, train_iterator, optimizer, criterion, HyperParams.clip)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')


def main():
    args = parse_args()
    min_word_count = args.min_word_count
    epochs = args.epochs
    # create new CPA model and train it
    conversations_file = args.conversations_file
    lines_file = args.lines_file

    # max_question_length = args.max_question_length
    # max_conversations = args.max_conversations

    print("Preparing data...")
    train_data, test_data = create_datasets(conversations_file, lines_file, 0.8)
    QFIELD.build_vocab(train_data, min_freq=min_word_count)
    AFIELD.build_vocab(train_data, min_freq=min_word_count)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    train_iterator, test_iterator = BucketIterator.splits(
        (train_data, test_data),
        batch_size=HyperParams.batch_size,
        device=device)

    input_dim = len(QFIELD.vocab)
    output_dim = len(AFIELD.vocab)

    enc = Encoder(input_dim, HyperParams.encoder_emb_dim, HyperParams.hidden_dim, HyperParams.enc_dropout)
    dec = Decoder(output_dim, HyperParams.decoder_emb_dim, HyperParams.hidden_dim, HyperParams.dec_dropout)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters:,} trainable parameters')

    if args.load:
        # read CPA model from file instead of training from scatch
        print(f'Loading CPA model from {args.load}')
        model.load_state_dict(torch.load(args.load))
    else:
        print("Training...")
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=AFIELD.vocab.stoi[AFIELD.pad_token])
        train_model(model, train_iterator, optimizer, criterion, epochs)
        if args.save:
            print(f'Saving model to {args.save}')
            torch.save(model.state_dict(), args.save)

    print("Starting ChatBot")
    # evaluate(model, test_iterator, criterion)
    test(model)


if __name__ == '__main__':
    main()
