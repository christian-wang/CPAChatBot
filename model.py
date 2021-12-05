import random
import torch
import hyperparams as hp
from typing import List, Tuple, Dict
from torch import nn as nn, optim
from torch.nn import functional
from tqdm import tqdm
from torchtext.legacy.vocab import Vocab
from attention import Attention
from preprocessing import prepare_training_batch, words_to_ints, normalize


class EncoderRNN(nn.Module):
    """
    Stacked bidirectional RNN with GRU and dropout
    """

    def __init__(self, embedding):
        super(EncoderRNN, self).__init__()
        self.embedding = embedding

        self.gru = nn.GRU(hp.HIDDEN_LAYER_DIM,
                          hp.HIDDEN_LAYER_DIM,
                          hp.ENCODER_LAYERS,
                          dropout=(0 if hp.ENCODER_LAYERS == 1 else hp.DROPOUT),
                          bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        """
        Encoder forward pass
        """
        embedded = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :hp.HIDDEN_LAYER_DIM] + outputs[:, :, hp.HIDDEN_LAYER_DIM:]
        return outputs, hidden


class DecoderRNN(nn.Module):
    """
    Stacked unidirectional RNN with GRU and dropout
    """

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
        self.attention = Attention.build(hp.ATTENTION_TYPE)

    def forward(self, input_step, last_hidden, encoder_outputs):
        """
        Forward pass through the decoder
        """
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
    """
    Greedy decoder.
    Computes argmax at every step of decoder to generate word.
    """

    def __init__(self, encoder, decoder, device, vocab):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = vocab[hp.SOS]

    def forward(self, input_seq, input_length, max_length):
        """
        Greedy decoder forward pass.
        """
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
    """
    ChatBot class. This class is responsible for training and evaluating the model.
    """

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
        """
        Returns a response given an input sentence.

        :param sentence: Input sentence.
        :param searcher: Decoder used to generate response.
        :return: Response sentence.
        """
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
        """
        Runs the ChatBot. Takes user input via command-line and prints responses.
        """
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

    def test(self, test_qa: List[List[List[str]]]) -> Dict[str, str]:
        results = dict()
        searcher = GreedySearchDecoder(self.encoder, self.decoder, self.device, self.vocab)
        for question, true_answer in test_qa:
            answer = self.evaluate(question, searcher)
            results[' '.join(question)] = ' '.join(answer)
        return results

    def log_likelihood(self, decoder_output: torch.Tensor, ground_truth: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate mean negative log likelihood of the target given the decoder output.
        Ignores the padding tokens.

        :param decoder_output: Decoder output tensor.
        :param ground_truth: Ground-truth answer tensor.
        :param mask: Mask tensor
        :return: Mean negative log likelihood.
        """
        cross_entropy = -torch.log(torch.gather(decoder_output, 1, ground_truth.view(-1, 1)).squeeze(1))
        loss = cross_entropy.masked_select(mask).mean()
        loss = loss.to(self.device)
        return loss

    def train_batch(self, batch: Tuple):
        """
        Train against a batch of question-answers.

        :param batch: 5-tuple of (question_batch, question_lengths,
                      answer_batch, answer_mask, max_answer_len)
        :return: Average loss.
        """
        question_batch, question_lengths, answer_batch, answer_mask, max_answer_len = batch
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        question_batch = question_batch.to(self.device)
        answer_batch = answer_batch.to(self.device)
        answer_mask = answer_mask.to(self.device)
        question_lengths = question_lengths.to("cpu")

        loss = 0
        losses = []
        n_totals = 0

        encoder_outputs, encoder_hidden = self.encoder(question_batch, question_lengths)

        sos_idx = self.vocab[hp.SOS]
        decoder_input = torch.LongTensor([[sos_idx for _ in range(hp.BATCH_SIZE)]])
        decoder_input = decoder_input.to(self.device)

        decoder_hidden = encoder_hidden[:hp.DECODER_LAYERS]

        # teacher forcing, use the ground-truth target as the next input if True
        use_teacher_forcing = True if random.random() < hp.TEACHER_FORCING else False

        for t in range(max_answer_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                          decoder_hidden,
                                                          encoder_outputs)
            if use_teacher_forcing:
                decoder_input = answer_batch[t].view(1, -1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(hp.BATCH_SIZE)]])
                decoder_input = decoder_input.to(self.device)
            mask_loss = self.log_likelihood(decoder_output, answer_batch[t], answer_mask[t])
            mask_sum = answer_mask[t].sum().item()
            loss += mask_loss
            losses.append(mask_loss.item() * mask_sum)
            n_totals += mask_sum
        loss.backward()

        nn.utils.clip_grad_norm_(self.encoder.parameters(), hp.CLIP)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), hp.CLIP)

        # adjust weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return sum(losses) / n_totals

    def train(self, save_path: str, prev_iteration: int = 0):
        """
        Train model.

        :param save_path: Path to save model.
        :param prev_iteration: Previous iteration number. 0 if new training.
        """
        start_iteration = prev_iteration + 1
        iterations = hp.ITERATIONS + 1 - start_iteration

        # ensure batches are sorted by answer length, speeds up training time greatly
        # self.question_answers.sort(key=lambda x: len(x[1]), reverse=False)
        for iteration in tqdm(range(start_iteration, hp.ITERATIONS + 1), total=iterations):
            qa_batch = random.choices(self.question_answers, k=hp.BATCH_SIZE)
            training_batch = prepare_training_batch(qa_batch, self.vocab)
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
