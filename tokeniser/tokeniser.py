import os
from collections import Counter
from nltk.stem import PorterStemmer
import re

script_dir = os.path.dirname(__file__)

special_tokens = [
    "<PAD>",
    "<UNK>",
    "<COMMA>",
    "<PERIOD>",
    "<QUOTE>",
]


class Tokeniser:
    def __init__(self, corpuses=None):
        self.min_freq = 10
        self.use_stemming = True
        self.word_counts = Counter()

        self.stemmer = PorterStemmer()

        self.vocab_mapping_path = os.path.join(script_dir, "vocab_mapping.txt")
        self.vocab_mapping = None
        self.inv_vocab_mapping = None

        self.special_char_re = re.compile(r'[^a-z0-9\s.,"]')

        self._initialise_vocab_mapping(corpuses)

    def _initialise_vocab_mapping(self, corpuses):
        # if corpuses is provided then we want to generate a new vocabulary
        if corpuses:
            print("Generating new vocabulary mapping...")
            self.vocab_mapping = self._generate_vocab_mapping(corpuses)
            self._save_vocab_mapping(self.vocab_mapping)
        # if corpus is not provided then use the pre-existing vocabulary
        else:
            print("Reading existing vocabulary mapping...")
            self.vocab_mapping = self._read_vocab_mapping()

        # generate the inverse mapping
        self.inv_vocab_mapping = {idx: word for word, idx in self.vocab_mapping.items()}

    def _generate_vocab_mapping(self, corpuses):
        if os.path.exists(
            os.path.join(script_dir, "../sources/normalised_corpuses.txt")
        ):
            with open(
                os.path.join(script_dir, "../sources/normalised_corpuses.txt"), "r"
            ) as f:
                print("Reading existing normalised corpus from file...")
                corpus_tokens = f.read().split()
        else:
            print("Normalising corpuses...")
            combined_corpus_tokens = []
            for idx, corpus in enumerate(corpuses):
                print(f"Normalising corpus {idx}...")
                corpus_tokens = self._normalise_text(corpus)
                combined_corpus_tokens.extend(corpus_tokens)

                with open(
                    os.path.join(
                        script_dir, "../sources/normalised_corpus_" + str(idx) + ".txt"
                    ),
                    "w",
                ) as f:
                    f.write(" ".join(corpus_tokens))

                print("Normalised corpus saved to file.")

            print("Saving combined normalised corpuses to file...")
            with open(
                os.path.join(script_dir, "../sources/normalised_corpuses.txt"), "w"
            ) as f:
                f.write(" ".join(combined_corpus_tokens))

            print("Combined normalised corpuses saved to file.")

        self.word_counts.update(corpus_tokens)

        frequent_tokens = {
            word for word, count in self.word_counts.items() if count >= self.min_freq
        }

        all_tokens = set(special_tokens).union(frequent_tokens)

        return {word: idx for idx, word in enumerate(sorted(all_tokens))}

    def _save_vocab_mapping(self, vocab_mapping):
        with open(self.vocab_mapping_path, "w") as f:
            for word in vocab_mapping:
                f.write(f"{word}\n")

    def _read_vocab_mapping(self):
        vocab_mapping = {}
        with open(self.vocab_mapping_path, "r") as f:
            for idx, word in enumerate(f):
                vocab_mapping[word.strip()] = idx
        return vocab_mapping

    def _normalise_text(self, text):
        text = text.lower()  # turn everything to lowercase
        text = self.special_char_re.sub(" ", text)  # remove special characters
        text = (
            text.replace(",", " <COMMA> ")
            .replace(".", " <PERIOD> ")
            .replace('"', " <QUOTE> ")
            .split()
        )

        if self.use_stemming:
            words = [
                self.stemmer.stem(word) if not word.startswith("<") else word
                for word in text
            ]

        return words

    def text_to_tokens(self, text):
        tokens = self._normalise_text(text)
        return [token if token in self.vocab_mapping else "<UNK>" for token in tokens]

    def text_to_token_ids(self, text):
        words = self.text_to_tokens(text)
        tokens = [self.vocab_mapping[word] for word in words]
        return tokens

    def tokens_to_token_ids(self, tokens):
        return [
            self.vocab_mapping[token] if token in self.vocab_mapping else "<UNK>"
            for token in tokens
        ]

    def token_ids_to_text(self, token_ids):
        words = [self.inv_vocab_mapping[token_id] for token_id in token_ids]
        text = " ".join(words)
        return text


if __name__ == "__main__":
    # use this to regenerate corpus files and vocab mapping

    try:
        with open(os.path.join(script_dir, "../sources/text8"), "r") as f:
            text8_corpus = f.read()
    except FileNotFoundError:
        print("text8 not found, please download it and save it to sources/text8")
        exit()

    try:
        with open(os.path.join(script_dir, "../sources/hn_title_corpus.txt"), "r") as f:
            hn_title_corpus = f.read()
    except FileNotFoundError:
        print(
            "hn_title_corpus not found, please download it and save it to sources/hn_title_corpus.txt"
        )
        exit()

    corpuses = [text8_corpus, hn_title_corpus]

    tokeniser = Tokeniser(corpuses=corpuses)

    text = "In this corpus there are words."

    tokens = tokeniser.text_to_tokens(text)
    token_ids = tokeniser.text_to_token_ids(text)
    reconstructed_text = tokeniser.token_ids_to_text(token_ids)
    print("Tokens:")
    print(tokens)
    print("Token IDs:")
    print(token_ids)
    print("Reconstructed text:")
    print(reconstructed_text)
