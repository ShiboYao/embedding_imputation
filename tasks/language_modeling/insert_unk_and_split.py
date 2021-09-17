### Replace the words in the corpus that are not in the pretrained embedding set with <unk>
### Split into train, val, test sets

import sys
vocab_path = sys.argv[1]
corpus_path = sys.argv[2]

def read_word_list(path):
    with open(path) as f:
        vocab = f.read().split('\n')[:-1]
        
    return vocab

def insert_unk_token(line, vocab):
    newline = []
    for word in line.split():
        if word in vocab:
            newline.append(word)
        else:
            newline.append('<unk>')
    newline = ' '.join(newline)
    return newline

with open(corpus_path, 'r') as f:
    lines = f.readlines()
        
vocab = read_word_list(vocab_path)

for i,line in enumerate(lines):
    lines[i] = insert_unk_token(line, vocab)

with open('./data/pubmed/train.txt', 'w') as f:
    for line in lines[:8000]:
        f.write(line + '\n')
with open('./data/pubmed/valid.txt', 'w') as f:
    for line in lines[8000:9000]:
        f.write(line + '\n')
with open('./data/pubmed/test.txt', 'w') as f:
    for line in lines[9000:]:
        f.write(line + '\n')
    
print('splitting is done.')