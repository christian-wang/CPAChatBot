import re
from typing import Dict, List, Union
from torchtext.legacy.data import Example, Dataset, Field


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


def create_datasets(conversations_file, lines_file, src_field, trg_field, split_ratios: Union[List, float] = 0.8):
    exchanges = load_exchanges(conversations_file)
    lines = load_dialogues(lines_file)
    examples = []

    fields = [('src', src_field), ('trg', trg_field)]

    for question_id, answer_id in exchanges:
        question = lines[question_id]
        answer = lines[answer_id]
        # if len(question) > 13:
        #     continue
        # answer = ' '.join(answer.split()[:11])
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
