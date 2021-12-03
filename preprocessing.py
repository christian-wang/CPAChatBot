import itertools
import torch
import spacy
import re
import random
from tqdm import tqdm
from collections import Counter
from typing import List, Dict, Tuple
from torchtext.legacy.vocab import Vocab
import hyperparams as hp

try:
    spacy_en = spacy.load('en_core_web_sm')
except OSError:
    print("Missing spacy language model, run \"python3 -m spacy download en_core_web_sm\".")
    print("Exiting...")
    exit(1)


def load_exchanges(file_name: str) -> List[List[str]]:
    """
    Return list of conversations from file_name

    :param file_name: path to file
    :return: List of [questionLineID, answerLineID] 2-lists
    """
    exchanges = []
    try:
        with open(file_name, encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f):
                conversation = line.strip().split(' +++$+++ ')[-1][1:-1].replace("'", "").split(", ")
                for i in range(len(conversation) - 1):
                    question_id = conversation[i]
                    answer_id = conversation[i + 1]
                    exchanges.append([question_id, answer_id])

    except FileNotFoundError:
        print(f"Conversations file not found at \"{file_name}\"")
        exit(1)

    return exchanges


def normalize(text: str) -> List[str]:
    """
    Tokenizes English text from a string into a list of strings
    """
    text = re.sub(r'[^!?.,\w\s]+', '', text)
    text = text.replace('...', ' ')
    return [tok.text.strip().lower() for tok in spacy_en.tokenizer(text) if tok.text.strip()]


def load_dialogues(file_name: str) -> Dict[str, List]:
    """
    Return a dictionary containing lineID: line mapping
    where each line is a list of word strings
    """
    dialogues = dict()
    try:
        with open(file_name, encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f):
                line_id, character_id, movie_id, character_name, text = line.rstrip('\n').split(' +++$+++ ')
                dialogue = normalize(text)
                dialogues[line_id] = dialogue
    except FileNotFoundError:
        print(f"Lines file not found at \"{file_name}\"")
        exit(1)
    return dialogues


def get_question_answers(dialogues: Dict[str, List], exchanges: List[List[str]]) -> Tuple[List[List[List[str]]], Vocab]:
    """
    Takes dialogues and exchanges and returns a list of question and answer pairs.
    Any questions or answers that are longer than MAX_SENTENCE_LENGTH are ignored.

    :param dialogues: dictionary of lineID: line mapping
    :param exchanges: list of [questionLineID, answerLineID] 2-lists
    :return: list of question and answer pairs
    """
    question_answers = []
    word_counts = dict()
    for question_id, answer_id in exchanges:
        question = dialogues[question_id]
        answer = dialogues[answer_id]

        if len(question) < hp.MAX_SENTENCE_LENGTH and len(answer) < hp.MAX_SENTENCE_LENGTH:
            question_answers.append([question, answer])
            for sentence in (question, answer):
                for word in sentence:
                    if word_counts.get(word) is None:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1
    print("Using {}/{} question-answer pairs.".format(len(question_answers), len(exchanges)))
    word_counter = Counter(word_counts)

    # TODO add vectors, look at the Vocab class for details
    vocab = Vocab(word_counter, vectors="glove.6B.300d", min_freq=hp.MIN_WORD_COUNT, specials=(hp.PAD, hp.SOS, hp.EOS, hp.UNK))
    return question_answers, vocab


def words_to_ints(vocab: Vocab, sentence: List) -> List[int]:
    """
    Converts a list of words to a list of word indexes
    and appends EOS token index
    """
    return [vocab[word] for word in sentence] + [vocab[hp.EOS]]


def words_to_vectors(vocab: Vocab, sentence: List) -> List[torch.Tensor]:
    """
    Converts a list of words to a list of word vectors
    and appends EOS vector
    """
    return [vocab.vectors[vocab[word]] for word in sentence] + [vocab.vectors[vocab[hp.EOS]]]


def pad_sentences(vectorized_sentences: List[List[torch.Tensor]], pad_vector: torch.Tensor) -> List[List[torch.Tensor]]:
    """
    Return padded indexed sentences

    :param vectorized_sentences:  List of sentences represented as lists of word indexes
    :param pad_vector: index of PAD token
    """
    return list(itertools.zip_longest(*vectorized_sentences, fillvalue=pad_vector))


def get_padding_matrix(padded_sentences: List[List[torch.Tensor]], pad_vector: torch.Tensor) -> List[List[int]]:
    """
    Returns a binary matrix (consisting of 1s and 0s) of
    shape (batch_size, max_sentence_length) that indicates
    whether a word is present in a sentence or not.

    :param padded_sentences: List of sentences represented as lists of word indexes
    :param pad_vector: padding word vector
    :return: binary matrix of shape (batch_size, max_sentence_length)
    """
    m = []
    for i, seq in enumerate(padded_sentences):
        m.append([])
        for token in seq:
            if token == pad_vector:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def prepare_question_batch(question_batch: List[List[str]], vocab: Vocab):
    """
    Converts list of questions to a tensor of word indices.
    Word indices are padded to the maximum sentence length.

    :param question_batch: list of questions
    :param vocab: Vocabulary object
    :return: padded tensor of word indices, lengths as 2D tensor
    """
    indices_batch = [words_to_vectors(vocab, question) for question in question_batch]
    lengths = torch.tensor([len(sentence) for sentence in indices_batch])
    padded_batch = torch.FloatTensor(pad_sentences(indices_batch, vocab.vectors[vocab[hp.PAD]]))
    return padded_batch, lengths


def prepare_answer_batch(answer_batch: List[List[str]], vocab: Vocab):
    """
    Converts list of answers to a tensor of word indices.
    Word indices are padded to the maximum sentence length.

    :param answer_batch: list of answers
    :param vocab: Vocabulary object
    :return: padded tensor of word indices, padding mask, max sentence length
    """
    indices_batch = [words_to_vectors(vocab, answer) for answer in answer_batch]
    max_answer_len = max([len(sentence) for sentence in indices_batch])
    padded_batch = pad_sentences(indices_batch, vocab.vectors[vocab[hp.PAD]])
    mask = torch.BoolTensor(get_padding_matrix(padded_batch, vocab.vectors[vocab[hp.PAD]]))
    padded_batch_tensor = torch.FloatTensor(padded_batch)
    return padded_batch_tensor, mask, max_answer_len


def prepare_training_batch(question_answer_batch: List[List[List[str]]], vocab: Vocab):
    """
    Returns batch of training data with padding and masking information.

    :param vocab: Vocabulary object.
    :param question_answer_batch: list of question and answer pairs.
    :return: padded tensor of word indices, question lengths as 2D tensor,
             padded tensor of answers, answer padding mask, max answer length
    """
    question_answer_batch.sort(key=lambda x: len(x[0]), reverse=True)
    question_batch: List[List[str]] = []
    answer_batch: List[List[str]] = []
    for question, answer in question_answer_batch:
        question_batch.append(question)
        answer_batch.append(answer)
    padded_question_batch, question_lengths = prepare_question_batch(question_batch, vocab)
    answer_batch, answer_mask, max_answer_len = prepare_answer_batch(answer_batch, vocab)
    return padded_question_batch, question_lengths, answer_batch, answer_mask, max_answer_len
