import random
import torch
import hyperparams as hp
from typing import List
from torch import nn as nn, optim
from torch.nn import functional
from tqdm import tqdm
from torchtext.legacy.vocab import Vocab
from attention import DotAttention
from preprocessing import get_training_batch, words_to_ints, normalize


class EncoderRNN(nn.Module):
    def __init__(self, embedding):
        super(EncoderRNN, self).__init__()
        self.embedding = embedding

        self.gru = nn.GRU(hp.HIDDEN_LAYER_DIM,
                          hp.HIDDEN_LAYER_DIM,
                          hp.ENCODER_LAYERS,
                          dropout=(0 if hp.ENCODER_LAYERS == 1 else hp.DROPOUT),
                          bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :hp.HIDDEN_LAYER_DIM] + outputs[:, :, hp.HIDDEN_LAYER_DIM:]
        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, embedding, output_size):
        super(DecoderRNN, self).__init__()

        self.output_size = output_size
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(hp.DROPOUT)
        self.gru = nn.GRU(hp.HIDDEN_LAYER_DIM,
                          hp.HIDDEN_LAYER_DIM,
                          hp.DECODER_LAYERS,
                          dropout=(0 if hp.DECODER_LAYERS == 1 else hp.DROPOUT))
        self.concat = nn.Linear(hp.HIDDEN_LAYER_DIM * 2, hp.HIDDEN_LAYER_DIM)
        self.out = nn.Linear(hp.HIDDEN_LAYER_DIM, output_size)

        self.attention = DotAttention()

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attention(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = functional.softmax(output, dim=1)
        return output, hidden


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, device, vocab):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = vocab[hp.SOS]

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

        decoder_hidden = encoder_hidden[:hp.DECODER_LAYERS]
        decoder_input = torch.ones(1, 1, device=self.device, dtype=torch.long) * self.sos_idx
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores


class CPAChatBot:
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN,
                 encoder_optimizer: optim.Optimizer, decoder_optimizer: optim.Optimizer,
                 embedding: nn.Embedding, vocab: Vocab,
                 question_answers: List[List[List[str]]], device: torch.device):
        self.encoder: EncoderRNN = encoder
        self.decoder: DecoderRNN = decoder
        self.encoder_optimizer: optim.Optimizer = encoder_optimizer
        self.decoder_optimizer: optim.Optimizer = decoder_optimizer
        self.embedding: nn.Embedding = embedding
        self.vocab: Vocab = vocab
        self.question_answers: List[List[List[str]]] = question_answers
        self.device: torch.device = device

    def evaluate(self, sentence: List[str], searcher: GreedySearchDecoder) -> List[str]:
        indexes_batch = [words_to_ints(self.vocab, sentence)]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        input_batch = input_batch.to(self.device)
        lengths = lengths.to("cpu")
        tokens, scores = searcher(input_batch, lengths, hp.MAX_SENTENCE_LENGTH)
        answer = []
        for token in tokens:
            word = self.vocab.itos[token.item()]
            if word not in {hp.EOS, hp.PAD}:
                answer.append(word)
        return answer

    def run(self):
        searcher = GreedySearchDecoder(self.encoder, self.decoder, self.device, self.vocab)
        while True:
            # Get input sentence
            user_input = input("You: ").lower()
            if user_input in {'q', 'quit', 'exit'}:
                print("Exiting...")
                break

            question = normalize(user_input)
            answer = self.evaluate(question, searcher)
            print('CPA:', ' '.join(answer))

    def log_likelihood(self, decoder_output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        cross_entropy = -torch.log(torch.gather(decoder_output, 1, target.view(-1, 1)).squeeze(1))
        loss = cross_entropy.masked_select(mask).mean()
        loss = loss.to(self.device)
        return loss

    def train_batch(self, batch):
        input_variable, lengths, target_variable, mask, max_target_len = batch
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_variable = input_variable.to(self.device)
        target_variable = target_variable.to(self.device)
        mask = mask.to(self.device)
        lengths = lengths.to("cpu")

        loss = 0
        losses = []
        n_totals = 0

        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)

        sos_idx = self.vocab[hp.SOS]
        decoder_input = torch.LongTensor([[sos_idx for _ in range(hp.BATCH_SIZE)]])
        decoder_input = decoder_input.to(self.device)

        decoder_hidden = encoder_hidden[:hp.DECODER_LAYERS]

        use_teacher_forcing = True if random.random() < hp.TEACHER_FORCING else False

        for t in range(max_target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                          decoder_hidden,
                                                          encoder_outputs)
            if use_teacher_forcing:
                decoder_input = target_variable[t].view(1, -1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(hp.BATCH_SIZE)]])
                decoder_input = decoder_input.to(self.device)
            mask_loss = self.log_likelihood(decoder_output, target_variable[t], mask[t])
            mask_sum = mask[t].sum().item()
            loss += mask_loss
            losses.append(mask_loss.item() * mask_sum)
            n_totals += mask_sum
        loss.backward()

        nn.utils.clip_grad_norm_(self.encoder.parameters(), hp.CLIP)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), hp.CLIP)

        # Adjust model weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return sum(losses) / n_totals

    def train(self, save_path, prev_iteration=0):
        training_batches = [
            get_training_batch([random.choice(self.question_answers) for _ in range(hp.BATCH_SIZE)], self.vocab)
            for _ in range(hp.ITERATIONS)]

        start_iteration = prev_iteration + 1

        for iteration in tqdm(range(start_iteration, hp.ITERATIONS + 1), total=hp.ITERATIONS + 1 - start_iteration):
            training_batch = training_batches[iteration - 1]

            loss = self.train_batch(training_batch)

            if save_path and iteration % hp.SAVE_EVERY == 0:
                torch.save({
                    'iteration': iteration,
                    'en': self.encoder.state_dict(),
                    'de': self.decoder.state_dict(),
                    'en_opt': self.encoder_optimizer.state_dict(),
                    'de_opt': self.decoder_optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': self.vocab.__dict__,
                    'embedding': self.embedding.state_dict()
                }, save_path)
