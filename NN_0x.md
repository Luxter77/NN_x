### NN_0 -> W_Embedding

Could be any of the document or text embeddings that are currently used like nomic-embed-text or ada-01 or any of those so long as they puke a dense embedding on the other side.
Then, they are used on a sliding window, over some input corpus, for the purposes of this document, the result of the sliding window will be referred as Te[]
Note that the embeding model used will be pre trained and frozen, its parameters are not to be messed with.

With hyperparameters: stride of the window, window size (ideally the lenght of the context of the embedidng model, but could be less if so choosed), embedding model (which one to use), sub stride
average (if a sub stride is picked to then be averaged the ouputs of that substride per each stride)