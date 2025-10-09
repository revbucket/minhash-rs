# A Primer on Minhash Theory and Engineering

Leveraging the MinHash algorithm for identifying and removing duplicates in large-scale text datasets has become a standard step in the curation of LLM pretraining datasets. We've been largely dissatisfied with the existing open-source deduplication frameworks, motivating the release of this package. One could easily just press the buttons and learn the tooling (as outlined in the essential_case_study), but it's also a bit helpful to understand a bit of the theory and engineering behind how an efficient large-scale minhash implementation might work. 

## The basics: Minhash in Theory

### The simplest minhash
First we define what we'll call a "duplicate", with respect to a text dataset. Minhash can be thought of as a set-similarity locality-sensitive hashing scheme: that is, if provided a collection of **sets**, we can determine which sets are "similar" in a Jaccard similarity sense. The Jaccard similarity of two sets, A, and B, is defined as the size of their intersection divided by the size of their union:
$$ J(A,B) := \frac{|A \cap B|}{|A \cup B|}.$$
Intuitively, if A=B, then this is 1.0; if A and B are disjoint, then this is 0.0; and if A and B share some elements, then this is somewhere in between. **We will define two sets as "fuzzy duplicates" if they have a Jaccard similarity greater than some threshold $T$.**

Consider these two hypothetical sets, A and B. Minhash provides a hash signature $S_A$, $S_B$ such that the probability of these two signatures being equal is equivalent to the Jaccard similarity of A and B:
$$ P[S_A=S_B] = J(A,B) $$
where we're hiding that the probability here refers to which specific hash function (from a large family of hash functions) we use to generate $S_A$ and $S_B$. 

Specifically the way that this works is to pick a hash function $h: X \rightarrow U$, which operates on elements of $A, B$,  mapping them into some universe that has a total ordering (say... integers). The minhash signature we'd get from set $A$ would be the minimum hash value of all its elements:
$$S_A := \min_{x\in A} h(x).$$

It is not too hard to see that this signature scheme satisfies the desired condition that $ P[S_A=S_B] = J(A,B) $. Assuming a sufficiently good hash function (read: collision-free), the only way for $S_A$ to equal $S_B$, would be if they share the minimum element: in other words, amongst all the elements in $A \cup B$, the minimum element lives in $A \cap B$, which happens with probability $J(A,B)$. Indeed, this is an if-and-only-if relationship: $S_A = S_B$ if and only if $argmin_{x\in A\cup B} \in A\cap B$. The reverse direction is trivial, and the forward direction follows by a quick contrapositive argument. 

This simple single-hash scheme leaves two open questions: 1) This seems very high variance -- is there a way to reduce the variance of this signal? 2) how does this work when, instead of having only two sets, we have a large collection of sets $\mathcal{X} := \{X_1, X_2, ..., X_n\}$?  Let's answer these in turn.


### Amplifying the signal
First we answer the question of variance. It's helpful to keep in mind that for a Bernoulli random variable with probability $p$, the variance is $p\cdot(1-p)$. For the purposes of identifying "fuzzy duplicates", or sets that are mostly similar, we can set a Jaccard similarity threshold, $T$: we care less about actually calculating the Jaccard similarity between any two sets, but instead about identifying with a high degree of confidence sets that two sets have Jaccard similarity $\geq T$. In our current schema, the probability of linking sets $A$ and $B$ can only be attained by our minhash signal, which fires with probability $J(A,B)$. It's helpful to consider graphs as follows with $J(A,B)$ on the x-axis, and the probability of a "minhash collision" on the y-axis. In the single-hash setting, this graph is the identity function. An ideal curve here would look like the Heaviside step function:
$$ P[\text{minhash collision}] = \begin{cases} 
      0 & J(A,B) < T \\
      1 & J(A,B) \geq T 
   \end{cases}.
$$
A standard way to amplify this would be to use multiple independent hash functions, and then report a minhash-collision only if all minhash signatures are equal. The probability that this occurs would be $J(A,B)^k$, which sharpens the curve, but also "pushes it to the right". To allow more control over the threshold, one consider minhashing in a 2-dimensional way. Suppose we have $m \times k$ independent hash functions. For any set, we can generate a signature for each of these hash functions, which we arrange in an $m \times k$ matrix, yielding a "signature matrix" for each set. We can say that any two sets have a minhash collision if for any $j \leq m$, the $j^{th}$ rows of their signature matrices are equal. Any one row is equal with probability $J(A,B)^k$, and the probability that at least one row of the $m$ rows is equal is 
$$P[\text{collision}] = 1 - (1-J(A,B)^k)^m.$$ Intuitively, this makes the curve less sharp while also shifting it slightly to the left. 

### Multidocument cases
Next we answer the question of how to handle multiple documents. Returning to the single-hash setting, if we had a collection $\mathcal{X} := \{X_1, X_2, ..., X_n\}$, and applied a single minhash function all of these, yielding signatures $\{S_1, S_2, ..., S_n\}$, we could partition the sets according to signature-equality. This suggests that any two elements within a group have sufficiently large Jaccard similarity. It's helpful to think of this from a graph perspective: if each set $X_i$ represents a node in a graph, then we can draw an edge between $X_i$ and $X_j$ if $S_i=S_j$. This yields a graph that can be thought of as a union of disjoint cliques. 

Extending this to the multi-hash function scheme above, where we've amplified the minhash algorithm to incorporate signature "matrices" and consider any two sets $X_i$ and $X_j$ if their $l^{th}$ signature rows are equal (for any $l$), then each row is effectively a hash-signature and produces a graph that is a union of disjoint cliques. Since this occurs for each row, we can generate a full network of "connected" nodes by taking the union of edges provided by each row. 

Since the Jaccard similarity is not transitive in the sense that if $J(A,B) \geq T$ and $J(B, C) \geq T$, it does not necessarily follow that $J(A,C) \geq T$, we run into some design issues here. If the ultimate goal is to 

### Fast hashing (warning: some math here)
Taking a brief diversion to discuss _how_ we hash in an efficient manner, let's assume that the contents of each set $X_i$ are integers within a universe $U$. Minhash requires that the hash function be taken from a min-wise independent family of hash function. This means that, given a collection of $k$ elements, the probability (over choice in hash function) that any one of them is the minimum hash-value is $\frac{1}{k}$. The ideal, and easiest form of this, would be to choose arbitrary permutations over $U$ as our hash functions. But writing such a permutation down would take approximately $O(U\log U)$ bits, and therefore be massively impractical. A much more reasonable family of hash functions would be to choose a linear congruential operator $h_{a,b}(x)$ defined as 
$$h_{a,b}(x) := ax + b \mod m$$ 
where a and m are coprime. 

Roughly speaking, this is because the multiplication by $a$, mod $m$ leads to a uniform permutation of the input space. In practice, we use the following hash family:
$$h_{a}(x) := MSB_{64}(ax \mod 2^128)$$
where $a$ is odd and $MSB_{64}$ takes the 64 most significant bits. Notice the lack of $b$, since we only care about relative ordering for minhash. We assert without proof that this provides a 2-universal hash function, i.e. for any distinct $x, y$, 
$$P_{a}[h_a(x) = h_a(y)] \leq 2^{-64}.$$
Further, one can argue that a 2-universal hash family with output range $M$ provides approximate min-wise independence, satisfying that for an y set $S$ with $|S|=k$ and $x \in S$, 
$$|P_h[h(x) = min_{x\in S}a(x)] - \frac{1}{k}| = O(\frac{k^2}{M}).$$

This is slightly off from the min-wise independence that is desired by minhash, but it is well-known that minhash is robust to imperfect hash families. The critical thing here is that we are able to compute the hashes required by the minhash algorithm with only a multiplication operator and a bitwise operation and still end up with a solid approximation.

### Applications: Minhash on text datasets
Rounding things off before we get into the engineering side of this theory, let's explain how we treat a collection of text strings as a collection of sets amenable for minhashing. For a given text string, it can be written as a sequence of tokens $\[x1, x2, ..., x_m \]$. We view this as a set by considering the ngram shingling of the sequence, i.e. $\{x_{1:n+1}, x_{2:n+2}, ..., x_{m-n:m}\}$. Then we define two text strings as being near duplicates if their sets of ngrams have a jaccard similarity of at least $T$. 

## Practical Algorithmics
With all the above theoretical background in hand, we are now equipped to discuss the steps required to implement a large-scale, easily distributable minhash algorithm. The key ideas are as follows:
- Each document in our corpus can be viewed as a set of ngrams.
- Minhash parameters can be tuned such that each document can be hashed into a matrix of signatures.
- In the graph-theoretic lens, documents are nodes and two documents have an edge if they have row-wise equality in at least one row of their signature matrices.
- Connected components can be computed from this large graph, where all documents within one connected component are "fuzzy duplicates".
- All documents can then be annotated with their connected-component ID, or only one node can be kept from each connected component.


This gives way to a multistep algorithm.
1. Compute the signature matrices for each document.
2. For each row-index, group all documents that have full row-wise equality and mark enough edges to create a graph with the proper connected components.
3. From this edge list, build a large union find data structure and compute the connected components
4. Use the connected components to either annotate or clean the corpus of duplicates.

We include some more specific implementation details about these steps below:

### Step 1: Computing Signature Matrices
Certainly we need to compute the signature matrix of each document in our corpus, but really we only need to maintain the ability to check row-wise (also referred to as band-wise, or bucket-wise) equality. For example, if our minhash parameters dictate the signature matrices are of size $m \times b$, we only need to keep fingerprints of the $m$ rows for each document. If we use the hash function described above, each row signature is $64\cdot b$ bits, but can be further compressed by hashing into a 16-byte integer. 

Further, since we will ultimately need to only compare band-signatures within a single band, it makes sense to store these signatures in a way that makes for easy grouping of bands. And since we need to check equality, to save on memory usage, it further makes sense to separate the band-signatures by their prefixes such that we have separate files for each (band_id, band_signature_prefix) chunk, each containing lists of (document_id, band_signature) bytestrings.

Hence, the hashes corresponding to the $X^{th}$ band of signatures, with the $Y^{th}$ prefix chunk, for the $Z^{th}$ slice of data are stored in filenames like:
`band_X/sigchunk_Y/pathchunk_Z.sig.bin`

Noting paralellism: this step can be done in an embarrassingly parallel fashion across many disjoints slices of data. It just matters that all band_id and signature prefix files for the entire corpus are grouped together before the next step. 

### Step 2: Computing Edges
Once we have signatures for all documents in the corpus, we can migrate and reorganize the signature files such that for any particular $(X, Y)$, all $Z's$ of the `band_X/sigchunk_Y/pathchunk_Z.sig.bin` files live on a single node. Then all of these can be loaded into memory and large lists of (document_id, band_signature) can be loaded into memory and sorted based on on their band_signature. Then this now-sorted list can be traversed and if any two adjacent list elements have the same band_signature, they are preserved as an edge (document_id1, document_id2). A more efficient way to represent this collection of edges would be to just represent each sequence of identical band signatures as a sequence of their nodes with a special token to separate sequences. For example, if we had edges (document1, document2), (document2, document3), (document3, document4), this could instead be represented as $[document1, document2, document3, document4, SPECIALID]$. This can then be stored in a file named like `edges.X.Y.bin`.

Noting parallelism: this step can be done in parallel across the band_id and signature prefix. The number of these documents is controlled both by the number of bands (`minhash_praams.num_buckets` in the config), and number of prefix groups (`en_params.num_sig_chunks`).

### Step 3: Gathering Connected Components
Once edges have been collected from signatures, we can instantiate the graph and compute the connected components. There is remarkably little novelty here: simply instantiate a dynamic union find data structure and loop through the edge files, adding edges from the edge files as needed. It's helpful that this union find data structure be implemented in a lock-free threadsafe way. We find it faster to explicitly compress all the paths before calling a global `find` operation to identify the connected-component identity of all nodes. 

We store, for every node in the graph, its `(document_id, connected_component_id, connected_component_size, connected_component_index)`

Noting parallelism: this step **cannot** be done in a multinode setting. Hence this step requires the greatest memory footprint. The output data however can be distributed across multiple output files, corresponding to which file each `document_id` lives in. 

### Step 4: Cleaning the data 
With a list of `(document_id, connected_component_id, connected_component_size, connected_component_index)`, it becomes trivial to either annotate or deduplicate the original data. We load the entire metadata above into memory and then loop through each data file. If annotating, we simply add the metadata to the original json. If deduplicating, we check to see if the document has `connected_component_index > 0`, and if so, we remove it. 

