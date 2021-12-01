import itertools
import torch
import spacy
from tqdm import tqdm
from collections import Counter
from typing import List, Dict, Tuple
from torchtext.legacy.vocab import Vocab
from hyperparams import MAX_SENTENCE_LENGTH, PAD, SOS, EOS, UNK, MIN_WORD_COUNT

spacy_en = spacy.load('en_core_web_sm')


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
    return [tok.text.strip() for tok in spacy_en.tokenizer(text) if tok.text.strip()]


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

        if len(question) < MAX_SENTENCE_LENGTH and len(answer) < MAX_SENTENCE_LENGTH:
            question_answers.append([question, answer])
            for sentence in (question, answer):
                for word in sentence:
                    if word_counts.get(word) is None:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1
    print("Using {}/{} question-answer pairs.", len(question_answers), len(exchanges))
    word_counter = Counter(word_counts)
    # TODO add vectors
    vocab = Vocab(word_counter, min_freq=MIN_WORD_COUNT, specials=(PAD, SOS, EOS, UNK))
    return question_answers, vocab


def words_to_ints(vocab: Vocab, sentence: List) -> List[int]:
    """
    Converts a list of words to a list of word indexes
    and appends EOS token index
    """
    return [vocab[word] for word in sentence] + [vocab[EOS]]


def pad_sentences(indexed_sentences: List[List[int]], pad_idx: int) -> List[List[int]]:
    """
    Return padded indexed sentences

    :param indexed_sentences:  List of sentences represented as lists of word indexes
    :param pad_idx: index of PAD token
    """
    return list(itertools.zip_longest(*indexed_sentences, fillvalue=pad_idx))


def get_padding_matrix(padded_sentences: List[List[int]], pad_idx) -> List[List[int]]:
    """
    Returns a binary matrix (consisting of 1s and 0s) of
    shape (batch_size, max_sentence_length) that indicates
    whether a word is present in a sentence or not.

    :param padded_sentences: List of sentences represented as lists of word indexes
    :param pad_idx: index of PAD token
    :return: binary matrix of shape (batch_size, max_sentence_length)
    """
    m = []
    for i, seq in enumerate(padded_sentences):
        m.append([])
        for token in seq:
            if token == pad_idx:
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
    indices_batch = [words_to_ints(vocab, question) for question in question_batch]
    lengths = torch.tensor([len(sentence) for sentence in indices_batch])
    padded_batch = torch.LongTensor(pad_sentences(indices_batch, vocab[PAD]))
    return padded_batch, lengths


def prepare_answer_batch(answer_batch: List[List[str]], vocab: Vocab):
    """
    Converts list of answers to a tensor of word indices.
    Word indices are padded to the maximum sentence length.

    :param answer_batch: list of answers
    :param vocab: Vocabulary object
    :return: padded tensor of word indices, padding mask, max sentence length
    """
    indices_batch = [words_to_ints(vocab, answer) for answer in answer_batch]
    max_answer_len = max([len(sentence) for sentence in indices_batch])
    padded_batch = pad_sentences(indices_batch, vocab[PAD])
    mask = torch.BoolTensor(get_padding_matrix(padded_batch, vocab[PAD]))
    padded_batch_tensor = torch.LongTensor(padded_batch)
    return padded_batch_tensor, mask, max_answer_len


def get_training_batch(question_answer_batch: List[List[List[str]]], vocab: Vocab):
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
