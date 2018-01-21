#!/usr/bin/env bash
rm -f jawiki-latest-text.txt
rm -f jawiki-latest-text-sentences.txt
rm -f jawiki-latest-text-tokens.txt
rm -f jawiki-latest-text-sentences-tokens.txt
rm -f ja-gensim.50d.data.model
rm -f ja-gensim.50d.data.txt
rm -rf *.npy
python3 generate_vectors.py --mecab
