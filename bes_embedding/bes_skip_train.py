import tqdm
import collections
import more_itertools
import requests
import wandb
import torch
import os
import builtins
import urllib.request

# 
# 
# 
# Check and download corpus files in same directory as this script

script_dir = os.path.dirname(os.path.abspath(__file__))

sources = {
    "text8": {
        "url": "https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8",
        "path": os.path.join(script_dir, "sources/text8"),
    },
    "hn_title_corpus": {
        "url": "https://huggingface.co/datasets/12v12v/mlx6-1/resolve/main/hn_title_corpus.txt",
        "path": os.path.join(script_dir, "sources/hn_title_corpus.txt"),
    },
}

# Ensure the sources directory exists
os.makedirs(os.path.join(script_dir, "sources"), exist_ok=True)

# For each source, download the file if it doesn't exist
for name, source in sources.items():
    if not os.path.exists(source["path"]):
        print(f"{name} not found, downloading now...")
        urllib.request.urlretrieve(source["url"], source["path"])
        print(f"{name} downloaded and saved to {source['path']}")



#
#
#
torch.manual_seed(42)

# Combine text in text8 file with hn_title_corpus file. Assuming you have both in same directory to use either or combine them
# with open("text8", "r") as f:
    # text8 = f.read()

with open("hn_title_corpus", "r") as f:
    hn_title_corpus = f.read()

# combined_text = text8 + " " + hn_title_corpus

combined_text = hn_title_corpus



#
#
# r = requests.get("https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8")
# with open("text8", "wb") as f: f.write(r.content)
# with open('text8') as f: text8: str = f.read()


#
#
#
def preprocess(text: str) -> list[str]:
  text = text.lower()
  text = text.replace('.',  ' <PERIOD> ')
  text = text.replace(',',  ' <COMMA> ')
  text = text.replace('"',  ' <QUOTATION_MARK> ')
  text = text.replace(';',  ' <SEMICOLON> ')
  text = text.replace('!',  ' <EXCLAMATION_MARK> ')
  text = text.replace('?',  ' <QUESTION_MARK> ')
  text = text.replace('(',  ' <LEFT_PAREN> ')
  text = text.replace(')',  ' <RIGHT_PAREN> ')
  text = text.replace('--', ' <HYPHENS> ')
  text = text.replace('?',  ' <QUESTION_MARK> ')
  text = text.replace(':',  ' <COLON> ')
  words = text.split()
  stats = collections.Counter(words)
  words = [word for word in words if stats[word] > 5]
  return words


#
#
#
corpus: list[str] = preprocess(combined_text)
print(type(corpus)) # <class 'list'>
print(len(corpus))  # New> 51,319,153 Old> 16,680,599
print(corpus[:7])   # ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse']


#
#
#
def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
  word_counts = collections.Counter(words)
  vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
  int_to_vocab = {ii+1: word for ii, word in enumerate(vocab)}
  int_to_vocab[0] = '<PAD>'
  vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
  return vocab_to_int, int_to_vocab


#
#
#
words_to_ids, ids_to_words = create_lookup_tables(corpus)
tokens = [words_to_ids[word] for word in corpus]
print(type(tokens)) # <class 'list'>
print(len(tokens))  # 51,319,153
print(tokens[:7])   # [11236, 7237, 24, 4, 652, 2, 2613]



#
#
#
print(ids_to_words[11236])        # anarchism
print(words_to_ids['anarchism']) # 11236
print(words_to_ids['have'])      # 54
print(len(words_to_ids))         # New> 136,458 Old>63,642


#
#
#
class SkipGramFoo(torch.nn.Module):
  def __init__(self, voc, emb, _):
    super().__init__()
    self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
    self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
    self.sig = torch.nn.Sigmoid()

  def forward(self, inpt, trgs, rand):
    emb = self.emb(inpt)
    ctx = self.ffw.weight[trgs]
    rnd = self.ffw.weight[rand]
    out = torch.bmm(ctx, emb.unsqueeze(-1)).squeeze()
    rnd = torch.bmm(rnd, emb.unsqueeze(-1)).squeeze()
    out = self.sig(out)
    rnd = self.sig(rnd)
    pst = -out.log().mean()
    ngt = -(1 - rnd + 10**(-3)).log().mean()
    return pst + ngt


#
#
#
args = (len(words_to_ids), 64, 2)
mFoo = SkipGramFoo(*args)
print('mFoo', sum(p.numel() for p in mFoo.parameters()))
opFoo = torch.optim.Adam(mFoo.parameters(), lr=0.01)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#
#
#
windows = list(more_itertools.windowed(tokens, 3))
inputs = [w[1] for w in windows]
targets = [[w[0], w[2]] for w in windows]
input_tensor = torch.LongTensor(inputs)
target_tensor = torch.LongTensor(targets)
dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)


#
#
#
wandb.init(project='mlx6-word2vec', name='mFoo')
mFoo.to(device)
for epoch in range(1):
  prgs = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
  for inpt, trgs in prgs:
    inpt, trgs = inpt.to(device), trgs.to(device)
    rand = torch.randint(0, len(words_to_ids), (inpt.size(0), 2)).to(device)
    opFoo.zero_grad()
    loss = mFoo(inpt, trgs, rand)
    loss.backward()
    opFoo.step()
    wandb.log({'loss': loss.item()})


#
#
#
print('Saving...')
torch.save(mFoo.state_dict(), './weights.pt')
print('Uploading...')
artifact = wandb.Artifact('model-weights', type='model')
artifact.add_file('./weights.pt')
wandb.log_artifact(artifact)
print('Done!')
wandb.finish()
