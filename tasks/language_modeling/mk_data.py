import string
import sys
import os


NUM_SAMPLES = int(sys.argv[1])
base_path = './data/sampled_{}'.format(NUM_SAMPLES)
vocab_path = os.path.join(base_path, 'pubmed_vocab_{}.txt'.format(NUM_SAMPLES))
corpus_path = os.path.join(base_path, 'pubmed_corpus_{}.txt'.format(NUM_SAMPLES))

### Samples the first N lines from pubmed texts, removes punctuation and saves vocabulary.

def sample_pubmed(NUM_SAMPLES):
    print('sampling the first {} lines...'.format(NUM_SAMPLES))
    
    if not os.path.exists(base_path): os.makedirs(base_path)
    if not os.path.exists(os.path.join(base_path, 'LSI_embeds')): os.makedirs(os.path.join(base_path, 'LSI_embeds'))
    if not os.path.exists(os.path.join(base_path, 'APPNP_embeds')): os.makedirs(os.path.join(base_path, 'APPNP_embeds'))
    if not os.path.exists(os.path.join(base_path, 'graphs')): os.makedirs(os.path.join(base_path, 'graphs'))
        
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

    with open(corpus_path, 'w') as f:
        f.writelines(lines)
    print('sampled corpus saved at ' + corpus_path)

    ######## Find and save unique tokens ########

    vocab = set()
    for line in lines:
        tokens = line.split()
        for token in tokens:
            if token not in vocab:
                vocab.add(token)

    print('{} unique words in corpus'.format(len(vocab)))

    with open(vocab_path, 'w') as f:
        for token in list(vocab):
            f.write(token + '\n')

    print('vocabulary saved at ' + vocab_path)
        
dic = {"glove": "./pretrained_embeds/glove.6B.200d.txt",
       "google": "./pretrained_embeds/GoogleNews-vectors-negative300.txt",
       "bioword": "./pretrained_embeds/BioWordVec_PubMed_MIMICIII_d200.txt",
       "fast": "./pretrained_embeds/wiki-news-300d-1M.vec"}
    
def read_word_list(path):
    with open(path) as f:
        vocab = f.read().split('\n')[:-1]
        
    return vocab

def read_embedding(name):
    with open(dic[name], 'r') as f:
        preembed = f.read().split('\n')
        if len(preembed[-1]) < 2:
            del preembed[-1]
    preembed = [e.split(' ', 1) for e in preembed]
    words = [e[0] for e in preembed]
    embed = [e[1] for e in preembed]
    return dict(zip(words,embed))


def build_data(names, pubmed_vocab_path):
    pubmed_words = set(read_word_list(pubmed_vocab_path))
    
    for name in names:
        embed_dict = read_embedding(name)
        word_set = set(embed_dict.keys())
        words = pubmed_words.intersection(word_set)
        embed = [w + ' ' + embed_dict[w] for w in words]

        with open(os.path.join(base_path, "{}_embeds.txt".format(name)), 'w') as f:
            f.write('\n'.join(embed))
            print(name+" saved.")

if __name__ == "__main__":
    sample_pubmed(NUM_SAMPLES)           
    build_data(dic.keys(), vocab_path)
    
    with open(corpus_path, 'r') as f:
        lines = f.readlines()
    
    n = len(lines)

    with open(os.path.join(base_path, 'train.txt'), 'w') as f:
        f.writelines(lines[:int(n*0.8)])
    with open(os.path.join(base_path, 'valid.txt'), 'w') as f:
        f.writelines(lines[int(n*0.8):int(n*0.9)])
    with open(os.path.join(base_path, 'test.txt'), 'w') as f:
        f.writelines(lines[int(n*0.9):])
        
    print('corpus split into train-val-test and saved.')