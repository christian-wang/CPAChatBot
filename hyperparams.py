PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'
UNK = '<unk>'

# maximum sentence length to consider
MAX_SENTENCE_LENGTH = 10

# minimum word count
MIN_WORD_COUNT = 3

# max norm for gradient clipping
CLIP = 50.0

# number of features in hidden state
HIDDEN_LAYER_DIM = 500

# number of RNNs to stack in encoder
ENCODER_LAYERS = 2

# number of RNNs to stack in decoder
DECODER_LAYERS = 2

# teacher forcing ratio
TEACHER_FORCING = 1.0

# learning rate
ENCODER_LEARNING_RATE = 0.0001

# decoder learning rate
DECODER_LEARNING_RATE = 0.0005

# number of training iterations
ITERATIONS = 4000

# save model each n iterations
SAVE_EVERY = 500

# attention type
ATTENTION_TYPE = 'dot'

# dropout rate
DROPOUT = 0.1

# batch size
BATCH_SIZE = 64
