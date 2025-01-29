import os
from hn_corpus.hn_corpus import build_hn_corpus
from tokeniser.tokeniser import Tokeniser

script_dir = os.path.dirname(os.path.abspath(__file__))

text8_corpus_path = os.path.join(script_dir, "sources/text8")
hn_title_corpus_path = os.path.join(script_dir, "sources/hn_title_corpus.txt")

if not os.path.exists(text8_corpus_path):
    raise FileNotFoundError("text8 corpus not found")

if not os.path.exists(hn_title_corpus_path):
    print("HN title corpus not found, building it now...")
    build_hn_corpus()

with open(text8_corpus_path, "r") as f:
    text8_corpus = f.read()

with open(hn_title_corpus_path, "r") as f:
    hn_title_corpus = f.read()

corpus = text8_corpus + hn_title_corpus

if not os.path.exists(os.path.join(script_dir, "./tokeniser/vocab_mapping.txt")):
    print("Vocab mapping not found, building it now...")
    tokeniser = Tokeniser(corpus=corpus, min_freq=10)
else:
    tokeniser = Tokeniser(min_freq=10)
