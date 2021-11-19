# open and load in the data sets of movie lines and movie conversations
lines = open('corpus/movie_lines.txt', encoding='utf-8', errors= 'ignore').read().split('\n')

conversations = open('corpus/movie_conversations.txt', encoding='utf-8', errors= 'ignore').read().split('\n')

exchange = []
for conversation in conversations:
    #should the slice not start at 0???
    exchange.append(conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(",","").split())


dialogue ={}
for line in lines:
    dialogue[line.split(' +++$+++ ')[0]] = line.split(' +++$+++ ')[-1]

questions =[]
answers = []

for conversation in exchange:
    for i in range(len(conversation)-1):
        questions.append(dialogue[conversation[i]])
        answers.append(dialogue[conversation[i+1]])


#delete variables we don't need anymore to save memory for training
del(lines, conversations, exchange, dialogue, line, conversation, i)


#making fixed length of questions less than 13, idk why lol

fixedLengthQ =[]
fixedLengthA = []
for i in range(len(questions)):
    if len(questions[i]) < 13:
        fixedLengthQ.append(questions[i])
        fixedLengthA.append(questions[i])

#cleaning text, change uppercase to lowercase, remove punctuation.
# Uses regular expressions to do so
import re
from sre_constants import CATEGORY_LINEBREAK
def clean_text(txt):
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

clean_questions =[]
clean_answers = []

#apply cleaning to questions and answers
for line in fixedLengthQ:
    clean_questions.append(clean_text(line))

for line in fixedLengthA:
    clean_answers.append(clean_text(line))

#make fixed length for answers as well now
for i in range(len(clean_answers)):
    clean_answers[i] = ' '.join(clean_answers[i].split()[:11])

del(answers, i , line, questions , fixedLengthA, fixedLengthQ)

clean_answers = clean_answers[:30000]
clean_questions = clean_questions[:30000]

#creating dictionary of word counts 
# can optimize this. TODO optimize this
word2count = {}
for line in clean_questions:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word]+=1

for line in clean_answers:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word]+=1

del(word, line)

#removing words that have low count to help improve training time, create vocab
threshold = 5
vocab ={}
word_number = 0

for word, count in word2count.items():
    if count >= threshold:
        vocab[word] = word_number
        word_number += 1

del(word2count, word, count, threshold, word_number)

#append start of sentence and end of sentence tags to answers
for i in range(len(clean_answers)):
    clean_answers[i] = '<SOS> ' + clean_answers[i] + ' <EOS>' 

# add token to vocab and give it a unique id
tokens =['<PAD>', '<EOS>', '<OUT>', '<SOS>']
x= len(vocab)
for token in tokens:
    vocab[token] = x
    x +=1
#padding is usually signified by 0 in decoder
# cameron is the word that holds the 0 id
# so we swap them
# TODO: optimize this part
vocab['cameron'] = vocab['<PAD>']
vocab['<PAD>'] = 0

del(tokens, x)

#inverse the dictionary
inv_vocab ={w:v for v, w in vocab.items()}



#making inputs for encoder
#doesnt handle unknow words good TODO: figure out how to handle unkown words
encoder_inp = []
for line in clean_questions:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])


    encoder_inp.append(lst)


#making inputs for decoder
#doesnt handle unknow words good TODO: figure out how to handle unkown words
decoder_inp = []
for line in clean_answers:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])


    decoder_inp.append(lst)

del(clean_answers, clean_questions, line, lst, word)


#paddding inputs
from keras.preprocessing.sequence import pad_sequences
encoder_inp = pad_sequences(encoder_inp, 13, padding='post', truncating='post')
decoder_inp = pad_sequences(decoder_inp, 13, padding='post', truncating='post')

#getting context of inputj
decoder_final_output =[]
for i in decoder_inp:
    decoder_final_output.append(i[1:])

decoder_final_output = pad_sequences(decoder_final_output, 13, padding='post', truncating='post')


from tensorflow.keras.utils import to_categorical
#converts data from 2d to 3d
decoder_final_output = to_categorical(decoder_final_output, len(vocab))