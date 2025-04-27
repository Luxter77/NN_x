### NN_2 -> LM_inv
Some decoder that can turn an embedding Te[n] into the text that originated it, such that

```
e = W_Embedding(x);
y = LM_inv(e)
assert x == y;
```
Ideally some form of language model, maybe large, that can be conditioned on the embedding and reproduce a piece of text likely to have generated it