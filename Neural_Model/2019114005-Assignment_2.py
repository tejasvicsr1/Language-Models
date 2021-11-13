import string
import torch
import re
from torch import nn
import torch.nn.functional as F
import numpy as np
import numpy as np

def _basic_english_normalize(line):

    temp = line.lower()
    regex = "[^\w\s]"
    temp = re.sub(r"[^\w\s]", '', temp)
    ans = temp.split()
    return ans

def one_hot_encoding(dataset, dimensions):

    one_hot_encoded = list()
    counter = 0
    for data in dataset:
        counter += 1
        embedding = []
        small_counter = 0
        for int in data:
            small_counter += 1
            word_embedding = np.zeros(dimensions)
            word_embedding[int] = 1
            embedding.append(word_embedding)
    ans = np.array(embedding)
    one_hot_encoded.append(ans)
    sol = np.array(one_hot_encoded)
    return sol

def get_tokens(sentences):
    
    counter = 0
    tokens = list()
    for line in sentences:
        counter += 1
        normalized_line = _basic_english_normalize(line)
        token_counter = 0
        for word in normalized_line:
            token_counter += 1
            tokens.append(word)

    return tokens

def create_mini_batches(arr, batch_size, seq_length):
    
    num_tokens_in_each_batch = batch_size * seq_length
    arr_len = len(arr)
    K = arr_len // num_tokens_in_each_batch
    split = num_tokens_in_each_batch * K
    arr = arr[:split].reshape((batch_size, -1))

    for i in range(0, arr.shape[1], seq_length):
        temp_size = i + seq_length
        x = arr[:, i:temp_size]
        check = 0
        y = np.zeros_like(x)
        try:
            check = 1
            y[:, :-1] = x[:, 1:]
            y[:, -1] = arr[:, temp_size]
        except IndexError:
            check = 2
            y[:, :-1] = x[:, 1:]
            y[:, -1] = arr[:, 0]
        yield x, y

data = 'brown.txt'
with open(data, 'r', encoding="utf8") as f:
    sentences = f.readlines()

a = 28946
b = 48947
c = 18946
test_set = sentences[a:b]
train_set = sentences[:c]
validation_set = sentences[c:a]

tokens = get_tokens(train_set)
vocab = sorted(list(set(tokens)))
print(len(sentences), len(train_set), len(validation_set), len(test_set))
vocab_size = len(vocab)
int2word = dict(enumerate(vocab))
print('Total Tokens: %d' % len(tokens))
word2int = {word:integer for integer, word in int2word.items()}
vocab_size = len(word2int)
encoded_sentence = np.array([word2int[word] for word in tokens])
print(encoded_sentence.shape)
print('Unique Tokens: %d' % vocab_size)
if(not torch.cuda.is_available()):
    print('No GPU available, training on CPU; consider making n_epochs very small.')
else:
    print('Training on GPU!')
train_on_gpu = torch.cuda.is_available()

class LanguageRNN(nn.Module):
    checker = 0
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, learn_rate=0.001):
        a_count = 0
        super().__init__()
        self.vocab = tokens
        self.int2word = dict(enumerate(self.vocab))
        self.n_hidden = n_hidden
        vocab_size = len(self.vocab)
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.learn_rate = learn_rate
        self.word2int = {word: int for int, word in self.int2word.items()}
        a_count = 0
        self.lstm = nn.LSTM(vocab_size, n_hidden, n_layers, dropout = drop_prob, batch_first = True)
        prob = drop_prob
        self.dropout = nn.Dropout(prob)
        checker = 1
        self.fc = nn.Linear(n_hidden, vocab_size)

    def forward(self, x, hidden):
        checker = 2
        last_output, last_hidden_state = self.lstm(x, hidden)
        final_output = self.dropout(last_output)
        temp = []
        final_output = final_output.contiguous().view(-1, self.n_hidden)
        temp.append(final_output)
        final_output = self.fc(final_output)

        return final_output, last_hidden_state

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data
        if (not train_on_gpu):
            temp_1 = weight.new(self.n_layers, batch_size, self.n_hidden).zero_()
            temp_2 = weight.new(self.n_layers, batch_size, self.n_hidden).zero_()
            hidden = (temp_1, temp_2)
        else:
            temp_1 = weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda()
            temp_2 = weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda()
            hidden = (temp_1, temp_2)
        checker = 3
        return hidden

a_len = len(test_set)

EPOCH_SIZE = 10
BATCH_SIZE = 10
SEQ_LENGTH = 20
LEARN_RATE = 0.001
GRADIENT_CLIP = 5
VAL_FRAC = 0.1
PRINT_FREQ = 5
n_hidden=512
n_layers=1
batch_size = 90
seq_length = 20
n_epochs = 2

def train(RNN, data, epochs = EPOCH_SIZE, batch_size = BATCH_SIZE, seq_length = SEQ_LENGTH, learn_rate = LEARN_RATE, gradient_clip = GRADIENT_CLIP, val_frac = VAL_FRAC, print_freq = PRINT_FREQ):
    checker = 0
    RNN.train()
    temp = learn_rate
    optimizer = torch.optim.Adam(RNN.parameters(), lr=temp)
    gpu_check = 0
    criterion = nn.CrossEntropyLoss()

    if(train_on_gpu):
        gpu_check = 1
        RNN.cuda()
    a = len(data)
    b = 1 - val_frac
    val_idx = int(a * b)
    data = data[:val_idx], 
    val_data = data[val_idx:]
    dimension = len(RNN.vocab)
    counter = 0

    for e in range(epochs):
        h = RNN.init_hidden(batch_size)
        is_gpu = 0
        for x, y in create_mini_batches(data, batch_size, seq_length):
            x = one_hot_encoding(x, dimension)
            counter = counter + 1
            inputs = torch.from_numpy(x).float()
            targets = torch.from_numpy(y).float()
            if (train_on_gpu):
                inputs = inputs.cuda()
                is_gpu = 1
                targets = targets.cuda()

            bc = counter % print_freq 
            h = tuple([each.data for each in h])
            RNN.zero_grad()
            temp = batch_size * seq_length
            output, h = RNN(inputs, h)
            loss = criterion(output, targets.view(temp).long())
            loss.backward()
            nn.utils.clip_grad_norm_(RNN.parameters(), gradient_clip)
            optimizer.step()

            if bc == 0:
                val_losses = []
                val_h = RNN.init_hidden(batch_size)
                RNN.eval()
                for x, y in create_mini_batches(val_data, batch_size, seq_length):
                    x = one_hot_encoding(x, dimension)
                    i = []
                    x = torch.from_numpy(x).float()
                    y = torch.from_numpy(y).float()
                    inputs = x
                    val_h = tuple([each.data for each in val_h])
                    targets = y

                    if (train_on_gpu):
                        inputs = inputs.cuda() 
                        targets = targets.cuda()
                    mult = batch_size * seq_length
                    output, val_h = RNN(inputs, val_h)
                    ch = targets.view(mult).long()
                    val_loss = criterion(output, ch)
                    val_losses.append(val_loss.item())

                RNN.train()
                # print("Epoch: {}/{}...".format(e+1, epochs), "Step: {}...".format(counter), "Loss: {:.4f}...".format(loss.item()), "Val Loss: {:.4f}".format(np.mean(val_losses)))


RNN = LanguageRNN(vocab, n_hidden, n_layers)
# print(RNN)
train(RNN, encoded_sentence, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, learn_rate=LEARN_RATE, print_freq=PRINT_FREQ//5)

model_name = 'rnn_1_layer_epoch_18000_data.net'
checkpoint = {}
checkpoint['n_hidden'] = RNN.n_hidden
checkpoint['n_layers'] = RNN.n_layers
checkpoint['state_dict'] = RNN.state_dict()
checkpoint['tokens'] = RNN.vocab
# checkpoint = {'n_hidden': RNN.n_hidden, 'n_layers': RNN.n_layers, 'state_dict': RNN.state_dict(), 'tokens': RNN.vocab}

with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)

N = None

def predict(RNN, current_word, hidden_state = N, k_most_probable_words = N):
    
    prob = torch.ones(vocab_size)
    temp = [RNN.word2int[current_word]]
    input = torch.tensor(temp).unsqueeze(1).long()

    if(train_on_gpu):
        gpu = 1
        input = input.cuda()

    hidden_state = tuple([each.data for each in hidden_state])
    gpu = 0
    output, hidden_state = RNN(input, hidden_state)
    ch = 1
    prediction = torch.nn.functional.softmax(output, dim = ch).data

    if train_on_gpu:
        gpu = 1
        prediction = prediction.cpu()

    temp = [RNN.word2int[current_word]]
    prediction = prediction.numpy()[0][torch.tensor(temp).item()]
    gpu = 0
    return prediction, hidden_state

def sample(RNN, size, prime="The sun is getting low", top_k=None):

    if(train_on_gpu):
        gpu = 1
        RNN.cuda()
    else:
        gpu = 0
        RNN.cpu()

    generated_text = prime.lower().split()
    RNN.eval()
    print(generated_text)
    h = RNN.init_hidden(1)
    s_len = len(generated_text) - 3
    sent_prob = 1

    for i in range(s_len):
        sums = i + 3
        prime_text = generated_text[i:sums]
        predicted_probability, h = predict(RNN, generated_text[sums], h, k_most_probable_words = top_k)
        sent_prob *= predicted_probability

    return sent_prob

model_name = 'rnn_1_layer_epoch_18000_data.net'

with open(model_name, 'rb') as f:
    isloaded = True
    checkpoint = torch.load(f)

VALUE = 1000
val_1 = checkpoint['tokens']
val_2 = checkpoint['n_hidden']
val_3 = checkpoint['n_layers']
loaded_RNN = LanguageRNN(val_1, n_hidden=val_2, n_layers=val_3)
example_sentence = 'The cat ran over the table.'
loaded_RNN.load_state_dict(checkpoint['state_dict'])

print(sample(loaded_RNN, VALUE, prime=example_sentence, top_k=50))