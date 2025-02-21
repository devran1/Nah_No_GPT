import numpy as np

text = """Once upon a time in a small, misty village nestled between towering mountains, there lived a young girl named Elara. She had always been fascinated by the stories of ancient legends that her grandmother told her by the flickering firelight. The most intriguing of these tales was about the Whispering Woods, a forest said to be alive with magic and secrets.

Elara had often gazed at the woods from her bedroom window, its emerald canopy swaying gently in the breeze. The villagers spoke of strange happenings there—mysterious lights that danced among the trees at night, and soft, melodic voices that could be heard if one listened closely enough. Most villagers stayed away, fearing what they did not understand, but Elara's curiosity only grew.

One evening, unable to resist any longer, Elara decided to venture into the Whispering Woods. She packed a small satchel with bread, cheese, and a flask of water, and as the sun dipped below the horizon, she slipped out of her house. The air was cool and the world was painted in shades of twilight, a perfect backdrop for her adventure.

As she stepped into the woods, a sense of wonder enveloped her. The trees towered above her, their trunks thick and gnarled, and the leaves rustled softly as if whispering secrets. Elara walked deeper into the forest, her heart racing with excitement and a hint of fear.

After walking for what felt like hours, she stumbled upon a clearing bathed in moonlight. In the center stood an ancient oak tree, its branches sprawling like arms reaching for the stars. To her astonishment, tiny lights flickered around the tree, illuminating the space with a soft glow. They danced in the air, swirling and twinkling like stars fallen to earth.

Entranced, Elara approached the tree. As she did, the lights coalesced into a form—a shimmering figure materialized before her. It was a spirit of the forest, ethereal and radiant, with eyes that sparkled like the night sky.

“Welcome, young one,” the spirit said, its voice"""

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = np.array(encode(text), dtype=np.int64)
n = int(0.9*len(data)) # first 90% will be train, rest validation data
train_data = data[:n]
val_data = data[n:]



class TokenEncoding:
  def __init__(self, vocab_size, d_model):
    self.embedding = np.random.randn(vocab_size, d_model) *0.002

  def __call__(self,x):
    return self.forward(x)

  def forward(self, token_ids):
    return self.embedding[token_ids]

  def backward(token_ids, embedding, d_out):

      # Initialize gradient for the embedding matrix with zeros
      d_embedding = np.zeros_like(embedding)

      # Accumulate gradients for each token ID
      for i, token_id in enumerate(token_ids.flat):  # Flatten to handle batch and sequence indices
          d_embedding[token_id] += d_out[i // d_out.shape[1], i % d_out.shape[1]]

      return d_embedding

class LayerNorm:
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = np.ones(dim)
    self.beta = np.zeros(dim)

  def __call__(self, x):
    return self.forward(x)

  def forward(self,x):
    self.x=x
    self.var = x.var(-1,  keepdims=True) # (B, T)
    self.mean=x.mean(-1, keepdims=True)
    self.norm = (x - self.mean) / np.sqrt(self.var) # (B, T, D)
    z = self.norm * self.gamma+ self.beta # (B, T, D)
    return z

  def backward(self, dz):
    dnorm = dz * self.gamma
    dvar = dnorm * (self.x - self.mean) * (-0.5) * (self.var + self.eps)**(-1.5)
    dmean = dnorm * (-1 / np.sqrt(self.var + self.eps)) + dvar * np.mean(self.x - self.mean, axis=-1, keepdims=True) * (-2 / self.x.shape[-1])
    dx = dnorm / np.sqrt(self.var + self.eps) + dvar * 2 * (self.x - self.mean) / self.x.shape[-1] + dmean / self.x.shape[-1]

    dgamma = np.sum(dz * self.norm, axis=(0, 1))
    dbeta = np.sum(dz, axis=(0, 1))

    return dx#, dgamma, dbeta

class Cross_entropy():
  def __init__(self):
    pass

  def  __call__(self,x,y):
    return self.forward(x,y)

  def forward(self,x,y):
    x = np.clip(x, 1e-8, 1 - 1e-8)
    #print("x",x.shape) #32,40
    #print("y",y.shape) #32
    
    return -np.sum(y * np.log(x), axis=1).mean() #right way
    #wrong way
    #return -np.sum(y @ np.log(x +1e-8))

  def backward(self,x,y):
    return x-y

cross_entropy=Cross_entropy()

class Relu():
	def __init__(self):
		pass
	def __call__(self,x):
		return self.forward(x)

	def forward(self,x):
		return np.maximum(x,0)

	def backward(self,x):
		return np.where(x>0, 1, 0)

class Softmax():
  def __init__(self,cross_entopy=False):
    self.cross_entropy=cross_entopy

  def  __call__(self,x):
    return self.forward(x)

  def forward(self, x):
    """
    #check for some errors
    if np.any(np.isnan(x)):
      print("x is nan")
    if np.any(np.isinf(x)):
      print("x is inf")
    """
    x = np.clip(x, -1e10, 1e10)

    # Apply numerically stable softmax
    x_max = np.max(x, axis=-1, keepdims=True)  # Find max for numerical stability
    exp_x = np.exp(x - x_max)
    #exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))

    self.out=exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return self.out

  def backward(self,x):
    if self.cross_entropy==False:
      s = self.out
      jacobian=np.diagflat(s) - np.outer(s, s)
      dz=np.sum(jacobian, axis=1).reshape(s.shape)
      return dz

    if self.cross_entropy:
      return self.out-y

softmax=Softmax(cross_entopy=False)

class Head():
  def __init__(self,head_size):

    self.key = np.random.randn(n_embd, head_size) *0.002 #, bias=False)
    self.query = np.random.randn(n_embd, head_size)*0.002#, bias=False)
    self.value = np.random.randn(n_embd, head_size)*0.002#, bias=False)

  def __call_(self,x):
    return self.forward(x)

  def forward(self,x):
    B,T,C = x.shape
    self.x=x

    k=np.dot(x, self.key)
    #print("key",k.shape)
    q=np.dot(x, self.query)
    v=np.dot(x, self.value)

    wei=np.tril((q @ k.transpose(0,2,1)))/np.sqrt(head_size) #att0
    #print("wei \n",wei[0],wei.shape)
    wei=np.where(wei == 0, float("-inf"),wei) #cannot talk to future tokens
    #print("wei \n",wei[0],wei.shape)
    wei= softmax(wei)
    #print("wei \n",wei[0],wei.shape)
    out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
    return out

class MultiHeadAttention():
  def __init__(self,num_heads, head_size):
    self.heads =[Head(head_size) for _ in range(num_heads)]
    self.proj = np.random.randn(head_size * num_heads, n_embd) *0.002

  def __call_(self,x):
    return self.forward(x)

  def forward(self,x):
    out = np.concatenate([h.forward(x) for h in self.heads], axis=-1)
    x=np.dot(out,self.proj)
    return  x

class FeedFoward():

    def __init__(self, n_embd):
        self.linear1=np.random.randn(n_embd, 4 * n_embd)*0.002
        self.relu=Relu()
        self.linear2=np.random.randn(4 * n_embd, n_embd)*0.002

    def __call__(self,x):
      return self.forward(x)

    def forward(self, x):
        self.x1=np.dot(x,self.linear1)
        self.x2=self.relu(self.x1)
        self.x3=np.dot(self.x2,self.linear2)
        return self.x3

class Block():
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa.forward(self.ln1(x))
        x = x + self.ffwd.forward(self.ln2(x))
        self.x=x
        return x

class GPTLanguageModel():

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = TokenEncoding(vocab_size, n_embd)

        self.position_embedding_table = TokenEncoding(block_size, n_embd)

        self.blocks = [Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = np.random.randn(n_embd, vocab_size) *0.002

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table.forward(np.arange(T)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)

        x=[h.forward(x) for h in self.blocks][0] #use the lat value

        x = self.ln_f(x) # (B,T,C) 
        logits=np.dot(x, self.lm_head)
        #logits = self.lm_head(x) # (B,T,vocab_size) #same 

        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            
            #add 1 to reshape otherwise it will be (B*T,) not same thing with (B*T,1) there will problems if you use "*" multiplications invross entropy.
            targets = targets.reshape(B*T,1)
            self.log = logits.reshape(B*T, C)
            loss = cross_entropy(logits, targets)
 
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(idx_cond) #self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            
            probs = softmax(logits) # (B, C)

            # sample from the distribution
            #idx_next = np.multinomial(probs, num_samples=1) # (B, 1)

            # Randomly choose an index for each row based on the probability distribution
            indices = np.array([np.random.choice(len(row), p=row) for row in probs])

            # Reshape to match your  output shape
            idx_next = indices.reshape(-1, 1)

            # append sampled index to the running sequence
            idx = np.concatenate((idx, idx_next), axis=1) # (B, T+1)

        return idx

# data loader in batches
def get_batch(split):
	# generate a small batch of data of inputs x and targets y
	data = train_data if split == 'train' else val_data

	ix = np.random.randint(0,len(data)-block_size,(batch_size,))

	x = np.stack([data[i:i+block_size] for i in ix])
	y = np.stack([data[i+1:i+block_size+1] for i in ix])

	return x, y

def estimate_loss(eval_iters):
    out = {}
    #model.eval()
    for split in ['train', 'val']:
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model.forward(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    #model.train()
    return out

max_iters, eval_interval=100,10
learning_rate=0.001
block_size, batch_size=8, 4
n_embd,head_size=8,4
n_layer, n_head=6,4

model = GPTLanguageModel()
m = model#.to(device)

# print the number of parameters in the model
#print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
#optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(eval_interval)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model.forward(xb, yb)
#    model.backward(xb)
#    optimizer.zero_grad(set_to_none=True)
#    optimizer.step()

# generate from the model
context = np.zeros((1, 1), dtype=int)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
