import itertools
import re
import unicodedata
from collections import Counter
from typing import List, Dict, Tuple
import torch
from torchtext.legacy.vocab import Vocab
from hyperparams import MAX_LENGTH, PAD, SOS, EOS, UNK, MIN_COUNT


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


def load_dialogues(file_name: str) -> Dict[str, List]:
    """
    Return a dictionary containing lineID: dialogue mapping
    where each value is a list of word strings
    """
    dialogues = dict()
    try:
        with open(file_name, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_id, character_id, movie_id, character_name, text = line.rstrip('\n').split(' +++$+++ ')
                dialogue = normalizeString(text)
                dialogues[line_id] = dialogue
    except FileNotFoundError:
        print(f"Lines file not found at \"{file_name}\"")
        exit(1)
    return dialogues


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s) -> List[str]:
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return [word.strip() for word in s.split(' ')]


def get_question_answers(dialogues: Dict[str, List], exchanges: List[List[str]]) -> Tuple[List[List[List[str]]], Vocab]:
    print("Start preparing training data ...")
    question_answers = []
    word_counts = dict()
    for question_id, answer_id in exchanges:
        question = dialogues[question_id]
        answer = dialogues[answer_id]

        if len(question) < MAX_LENGTH and len(answer) < MAX_LENGTH:
            question_answers.append([question, answer])
            for sentence in (question, answer):
                for word in sentence:
                    if word_counts.get(word) is None:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1
    word_counter = Counter(word_counts)

    # TODO add vectors
    vocab = Vocab(word_counter, min_freq=MIN_COUNT, specials=(PAD, SOS, EOS, UNK))
    return question_answers, vocab


def trimRareWords(vocab: Vocab, question_answers: List[List[List[str]]]):
    # Trim words used under the MIN_COUNT from the voc
    # Filter out pairs with trimmed words
    keep_pairs = []
    for question, answer in question_answers:
        for word in question:
            if vocab[word] == vocab.unk_index:
                break
        else:
            for word in answer:
                if vocab[word] == vocab.unk_index:
                    break
            else:
                keep_pairs.append([question, answer])

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(question_answers), len(keep_pairs),
                                                                len(keep_pairs) / len(question_answers)))
    return keep_pairs


def indexesFromSentence(vocab: Vocab, sentence: List) -> List[int]:
    return [vocab[word] for word in sentence] + [vocab[EOS]]


def zeroPadding(indexed_sentences: List[List[int]], pad_idx: int) -> List[List[int]]:
    """
    Return padded indexed sentences

    :param indexed_sentences:  List of sentences represented as lists of word indexes
    :param pad_idx: index of PAD token
    """
    return list(itertools.zip_longest(*indexed_sentences, fillvalue=pad_idx))


def binaryMatrix(padded_sentences: List[List[int]], pad_idx):
    m = []
    for i, seq in enumerate(padded_sentences):
        m.append([])
        for token in seq:
            if token == pad_idx:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def inputVar(questions: List[List[str]], vocab: Vocab):
    indexes_batch = [indexesFromSentence(vocab, question) for question in questions]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch, vocab[PAD])
    padVar = torch.LongTensor(padList)
    return padVar, lengths


def outputVar(answers: List[List[str]], vocab: Vocab):
    indexes_batch = [indexesFromSentence(vocab, answer) for answer in answers]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch, vocab[PAD])
    mask = torch.BoolTensor(binaryMatrix(padList, vocab[PAD]))
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


def batch2TrainData(vocab: Vocab, pair_batch: List[List[List[str]]]):
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch: List[List[str]] = []
    output_batch: List[List[str]] = []
    for question, answer in pair_batch:
        input_batch.append(question)
        output_batch.append(answer)
    inp, lengths = inputVar(input_batch, vocab)
    output, mask, max_target_len = outputVar(output_batch, vocab)
    return inp, lengths, output, mask, max_target_len
