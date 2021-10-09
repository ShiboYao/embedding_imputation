### Samples the first N lines from pubmed texts, removes punctuation and saves vocabulary.

import string
import sys
import os

def sample_pubmed(NUM_SAMPLES)

    print('sampling the first {} lines...'.format(NUM_SAMPLES))
    base_path = './data/sampled_{}'.format(NUM_SAMPLES)
    if not os.path.exists(base_path): os.makedirs(base_path)

    #Sample the first few lines, remove punctuation only
    lines = []
    with open('./data/pubmed_sentence_nltk.txt', 'r') as f:
        for i in range(NUM_SAMPLES):
            line = f.readline()
            #in the provided pytorch example, they keep the punctuation, so putting space around punctuation. 
            #line = line.translate(str.maketrans({key: " {} ".format(key) for key in string.punctuation}))
            #replace punctuation with white space
            #line = line.translate(str.maketrans('', '', string.punctuation))
            lines.append(line)

    save_path = os.path.join(base_path, 'pubmed_corpus_{}.txt'.format(NUM_SAMPLES))
    with open(save_path, 'w') as f:
        f.writelines(lines)

    print('sampled corpus saved at ' + save_path)


    ######## Find and save unique tokens ########

    vocab = set()
    for line in lines:
        tokens = line.split()
        for token in tokens:
            if token not in vocab:
                vocab.add(token)

    print('{} unique words in corpus'.format(len(vocab)))

    save_path = os.path.join(base_path, 'pubmed_vocab_{}.txt'.format(NUM_SAMPLES))

    with open(save_path, 'w') as f:
        for token in list(vocab):
            f.write(token + '\n')

    print('vocabulary saved at ' + save_path)