import re
from typing import List, Dict
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

TAGS = {'<PAD>', '<EOS>', '<OUT>', '<SOS>'}


def load_conversations(file_name: str) -> List[List[str]]:
    """
    Return list of conversations from file_name

    :param file_name: path to file
    :return: List of lists containing line ids
    """
    conversations = []
    try:
        with open(file_name, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_ids = line.strip().split(' +++$+++ ')[-1][1:-1].replace("'", "").split(", ")
                conversations.append(line_ids)
    except FileNotFoundError:
        print(f"Conversations file not found at \"{file_name}\"")
        exit(1)
    return conversations


def load_dialogues(file_name: str) -> Dict[str, str]:
    dialogues = dict()
    try:
        with open(file_name, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_id, character_id, movie_id, character_name, text = line.rstrip('\n').split(' +++$+++ ')
                dialogues[line_id] = text
    except FileNotFoundError:
        print(f"Lines file not found at \"{file_name}\"")
        exit(1)
    return dialogues


def clean_text(txt):
    # cleaning text, change uppercase to lowercase, remove punctuation.
    # Uses regular expressions to do so
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt


def get_question_answers(conversations, lines, max_question_len=13, max_conversations=30000):
    questions_answers = []
    for conversation in conversations:
        for i in range(len(conversation) - 1):

            question = clean_text(lines[conversation[i]])
            # skip any questions that are too long
            if len(question.split()) >= max_question_len:
                continue

            # truncate answer to be first 11 words
            answer = ' '.join(clean_text(lines[conversation[i + 1]]).split()[:11])

            # add SOS and EOS tags to answer
            answer = '<SOS> ' + answer + ' <EOS>'
            # trunc_answer = ' '.join(answer.split()[:11])
            questions_answers.append([question, answer])

    # use first max_conversations questions and answers
    return questions_answers[:max_conversations]


# creating dictionary of word counts
def get_word_counts(question_answers):

    word_counts = {}
    for question, answer in question_answers:
        for word in question.split():
            if word in TAGS:
                continue
            elif word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1
        for word in answer.split():
            if word in TAGS:
                continue
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1
    return word_counts


def create_vocab(word2count, threshold):

    # removing words that have low count to help improve training time, create vocab
    vocab = dict()
    word_number = 0

    for word, count in word2count.items():
        if count >= threshold:
            vocab[word] = word_number
            word_number += 1

    # add tags to vocabulary
    token_id = len(vocab)
    for token in TAGS:
        vocab[token] = token_id
        token_id += 1

    # invert the vocabulary
    inv_vocab = {w: v for v, w in vocab.items()}

    # ensure vocab['<PAD>'] = 0
    old_pad = vocab['<PAD>']
    old_zero = inv_vocab[0]
    vocab['<PAD>'] = 0
    vocab[old_zero] = old_pad
    inv_vocab[old_pad] = old_zero
    inv_vocab[0] = '<PAD>'

    return vocab, inv_vocab


def create_inputs(question_answers, vocab):

    # making inputs for encoder
    # doesnt handle unknown words good. figure out how to handle unknown words
    encoder_inp = []
    decoder_inp = []

    for question, answer in question_answers:
        q_list = []
        a_list = []

        for word in question.split():
            if word not in vocab:
                q_list.append(vocab['<OUT>'])
            else:
                q_list.append(vocab[word])
        encoder_inp.append(q_list)

        for word in answer.split():
            if word not in vocab:
                a_list.append(vocab['<OUT>'])
            else:
                a_list.append(vocab[word])

        decoder_inp.append(a_list)

    encoder_inp = pad_sequences(encoder_inp, 13, padding='post', truncating='post')
    decoder_inp = pad_sequences(decoder_inp, 13, padding='post', truncating='post')
    return encoder_inp, decoder_inp


def get_decoder_final_output(decoder_inp, vocab):

    # getting context of input
    decoder_final_output = []
    for i in decoder_inp:
        decoder_final_output.append(i[1:])

    decoder_final_output = pad_sequences(decoder_final_output, 13, padding='post', truncating='post')

    # converts data from 2d to 3d
    decoder_final_output = to_categorical(decoder_final_output, len(vocab))
    return decoder_final_output
