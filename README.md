# Japanese Word2Vec

## About

Word2vec (word to vectors) approach for Japanese language using Gensim (Deep Learning skip-gram and CBOW models). The model is trained on the *Japanese version of Wikipedia* available at https://dumps.wikimedia.org/jawiki/latest/.

<i>Definition: Word2vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2vec takes as its input a large corpus of text and produces a high-dimensional space (typically of several hundred dimensions), with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located in close proximity to one another in the space.</i>

More information about the word2vec properties is available here: http://nlp.stanford.edu/projects/glove/

## Usage

Generating the vectors from a wikipedia dump takes about 2~3 hours on a Core i5, with the default parameters.
```
git clone https://github.com/philipperemy/japanese-word-to-vectors.git
cd japanese-word-to-vectors
pip install -r requirements.txt
python generate_vectors.py
```

### Convert Wiki dump to text
The first step is to extract the text and the sentences of the dump. It is done in this function:

```
INPUT_FILENAME = 'jawiki-latest-pages-articles.xml.bz2'
JA_WIKI_TEXT_FILENAME = 'jawiki-latest-text.txt'
JA_WIKI_SENTENCES_FILENAME = 'jawiki-latest-text-sentences.txt'
process_wiki_to_text(INPUT_FILENAME, JA_WIKI_TEXT_FILENAME, JA_WIKI_SENTENCES_FILENAME)
```

The output is two files `JA_WIKI_TEXT_FILENAME` whose content looks like:
`trebuchet msフォントアンパサンドとはを意味する記号である`
where each line corresponds to an article.

and `JA_WIKI_SENTENCES_FILENAME` where each line corresponds to a sentence or chunk of words in the text. This file will not be used in the word2vec algorithm but can be useful to train a sentence to vec (named skip-thoughts, available here https://github.com/ryankiros/skip-thoughts/).

### Tokenize the text

Tokenizing means adding spaces to the compacted text we have. We try to find out the words. For this, we use the library called `TinySegmenter3` (https://pypi.python.org/pypi/tinysegmenter3/0.1.0) which is able to tokenize japanese corpus with more than 95 percent accuracy (source: http://lilyx.net/tinysegmenter-in-python/).

The output is `JA_WIKI_TEXT_TOKENS_FILENAME`. It looks like this:
`trebuchet   ms   フォント   アンパサンド   と は   を 意味 する`

Finally the (Gensim)[https://radimrehurek.com/gensim/] library is used to perform the word2vec algorithm with the parameters:
- size of 50 (dimensionality of the feature vectors)
- window of 5 (maximum distance between the current and predicted word within a sentence)
- min count of 5 (ignore all words with total frequency lower than this)
- iter of 5 (number of iterations or epochs over the corpus)
- number of workers equal to number of cores

While training, the console output looks like:
```
2016-09-04 02:54:38,354 : INFO : PROGRESS: at 99.74% examples, 482630 words/s, in_qsize 5, out_qsize 4
2016-09-04 02:54:39,346 : INFO : PROGRESS: at 99.82% examples, 482644 words/s, in_qsize 7, out_qsize 0
2016-09-04 02:54:40,356 : INFO : PROGRESS: at 99.90% examples, 482643 words/s, in_qsize 7, out_qsize 1
2016-09-04 02:54:41,390 : INFO : PROGRESS: at 99.98% examples, 482630 words/s, in_qsize 8, out_qsize 0
```
Once it's finished, 4 new files are generated:
- ja-gensim.50d.data.model
This file contains the model in the binary format. Use `model = Word2Vec.load(fname)` to get your word2vec model.
- ja-gensim.50d.data.txt
This file contains the model in the text format. Can be used in any other python script without `Gensim` library!
- ja-gensim.50d.data.model.syn0.npy
File generated automatically. Contains some numpy arrays.
- ja-gensim.50d.data.model.syn1neg.npy
File generated automatically. Contains some numpy arrays.

Finally, let's inspect `ja-gensim.50d.data.txt`

```
の 0.128774 3.631298 -3.058414 -0.434418 -0.300449 -1.211774 0.608027 -5.561740 -1.186208 -0.035129 1.709353 1.252130 -3.849393 0.390795 4.260262 0.209959 2.316592 -2.880473 -0.427741 -1.335913 4.500565 0.556813 0.585122 -0.739895 1.034633 3.786435 -1.032835 -5.697092 1.436553 -1.689847 -4.953261 -3.883135 1.730590 -3.211419 -2.154781 -1.915586 -0.283341 0.332927 -2.281737 0.440092 1.535507 0.925073 -4.101060 0.634421 -4.230011 -0.313288 -3.955676 0.009256 2.931253 -0.500217
に -2.019490 4.359702 -1.845176 -2.663986 1.774256 0.147722 1.484422 -2.984465 2.262582 -0.861214 0.804603 1.007627 -4.322638 -0.173283 2.905254 0.803300 2.850667 -3.859382 -0.214240 -1.914028 5.640825 -0.139551 0.243700 -3.234274 1.844652 6.613075 -2.586612 -7.520448 4.413483 -3.270162 -2.952101 -2.278936 7.161888 -6.830038 -2.042799 -0.559094 -2.270651 2.744259 -2.250800 0.269468 -0.153715 3.831476 -2.068467 1.833452 -4.605278 3.756418 -4.275790 1.822912 1.606565 -2.918230
は 0.296134 4.136690 -3.184480 -0.817397 0.555022 -1.181827 0.933714 -4.486689 -0.429983 0.427427 0.089208 1.415648 -2.763912 1.310283 5.143843 1.778646 2.280496 -4.852800 -1.581973 -1.364721 3.240205 1.227000 0.931791 -2.009395 1.856946 3.401864 -1.741597 -6.626904 -0.016503 -3.313225 -2.302027 -3.208004 4.541845 -4.704424 -2.073442 -1.192726 0.880771 -1.584695 0.450757 1.645549 1.212130 1.006536 -3.576060 0.142494 -4.799853 0.906162 -3.141263 1.762820 2.482034 -1.188599
```
Here we can see the vectors for `の`, `に` and `は`. If we go deeper, we can see longer words such as `文献`. The size of the vocabulary is the number of lines of this file (one line equals one word and its vector representation).

`wc -l ja-gensim.50d.data.txt` yields `1200627` words.


## References
- Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
-	Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.
