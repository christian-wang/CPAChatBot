PAD = '<pad>'
SOS = '<sos>'
EOS = '<eos>'
UNK = '<unk>'

# Preprocessing settings
MAX_SENTENCE_LENGTH = 10  # Maximum sentence length to consider
MIN_WORD_COUNT = 3  # Minimum word count threshold for trimming


CLIP = 50.0

# Learning Hyperparameters
TEACHER_FORCING = 1.0
LEARNING_RATE = 0.0001
DECODER_LEARNING_RATIO = 5.0
ITERATIONS = 4000
SAVE_EVERY = 500
ATTENTION_TYPE = 'dot'
HIDDEN_LAYER_DIM = 500
ENCODER_LAYERS = 2
DECODER_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 64
