import sys
vocab_path = sys.argv[1]
embed_set = sys.argv[2] #glove or bioword

if embed_set == 'glove':
    dic = {"glove": "./pretrained_embeds/glove.6B.100d.txt"}
elif embed_set == 'bioword': 
    dic = {"bioword":"./pretrained_embeds/BioWordVec_PubMed_MIMICIII_d200.txt"}
else:
    raise ValueError('wrong pretrained embedding name')
    
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


def build_data(names):
    embed = [read_embedding(n) for n in names]
    words = set(read_word_list(vocab_path))
    word_sets = [set(e.keys()) for e in embed]
    word_sets.append(words)
    words = set.intersection(*word_sets)
    with open("./data/word_list_{}.txt".format(embed_set), 'w') as f:
        f.write('\n'.join(words))
        print("Word list saved.")
    embed = [[e[w] for w in words] for e in embed]
    for i in range(len(names)):
        with open("./data/{}_embeds.txt".format(names[i]), 'w') as f:
            f.write('\n'.join(embed[i]))
            print(names[i]+" saved.")


if __name__ == "__main__":
    build_data(list(dic.keys()))