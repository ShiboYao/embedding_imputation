# Language Modeling Task


## Download pretrained embeddings

First, you will need to download and place the pretrained embeddings into the `pretrained_embeds` folder.

*[GloVe](https://nlp.stanford.edu/projects/glove/): glove.6B.zip

*[Word2Vec](https://code.google.com/archive/p/word2vec/): 

*[FastText](https://fasttext.cc/docs/en/english-vectors.html):(wiki-news-300d-1M.vec.zip)

*[BioWordVec](https://github.com/ncbi-nlp/BioSentVec): 

You can use `convert_bin_to_txt.py` to convert .bin files to .txt files for BioWordVec and FastText

Then you need to download the preprocessed PubMed texts from [here] https://github.com/ncbi-nlp/bluebert and place them under the `data` folder.

Your directory should look like:

```
.
└───pretrained_embeds/
|   |   glove.6B.200d.txt
|   |   GoogleNews-vectors-negative300.txt
|   |   wiki-news-300d-1M.vec
|   |   BioWordVec_PubMed_MIMICIII_d200.txt
└───data/
|   |   pubmed_sentence_nltk.txt

```


## Preprocessing

First, run `python mk_data.py 20000`. This will sample the first 20,000 lines from the preprocessed PubMed texts, save the list of unique words and split into train-valid-test sets and create necessary embedding files. You can change 20,000 to sample a different number of lines.

Then, to create the graphs, run `python save_graphs.py`

## Imputation

To impute the missing embeddings you need to run:

* LSI: `python imputation_lsi.py <base> bioword <delta> ./data/sampled_20000/LSI_embeds <seed> <corpus_size>`
* GCN: `python train_gcn.py <base> --aff=bioword --seed=<seed> --corpus_size=<corpus_size>`
* APPNP: `python train_appnp.py --base=<base> --aff=bioword --p=<p> --seed=<seed> --corpus_size=<corpus_size>`

After the imputation, the embeddings will be saved to be used in the LSTM model.

Replace:
* base: "google", "glove" or "fast"
* delta: an integer (8)
* seed: an integer to set seed (1)
* corpus_size: number of sampled lines (20000)
* p: number of iterations in APPNP (10)
* alpha: as defined in APPNP (0.1)

with the desired values.

You can also utilize `impute_all.sh` to impute all at the same time.

## Training an LSTM model

* BASE: `python main.py --embed_path=data/sampled_<corpus_size>/<base>_embeds.txt --base=<base> --cuda`

* LSI: `python main.py --embed_path=data/sampled_<corpus_size>/LSI_embeds/LSI_<base>_embeds_<delta>_<seed>.txt --base=<base> --cuda`

* GCN: `python main.py --embed_path=data/sampled_<corpus_size>/GCN_embeds/GCN_<base>_embeds_<delta>_<seed>.txt --base=<base> --cuda`

* APPNP: `python main.py --embed_path=data/sampled_<corpus_size>/APPNP_embeds/APPNP_<base>_embeds_<delta>_<alpha>_<p>_<seed>.txt --base=<base> --cuda`

You can also use `run_main.sh` to run all of them at the same time. Replace the values in brackets as desired

Replace:
* base: "google", "glove" or "fast"
* delta: an integer (8)
* seed: an integer to set seed (1)
* corpus_size: number of sampled lines (20000)
* p: number of iterations in APPNP (10)
* alpha: as defined in APPNP (0.1)

omit `--cuda` if you want to run using CPU.