Equations from Wu, Shapiro & Cotterelll (2018)

**Embeddings**  
$\mathbf{e}^{(enc)}: \Sigma_x \to \mathbb{R}^{d_e}$ where embedding size is $d_e \in \{50, 100, 200, 300\}$  

$\mathbf{e}^{(dec)}: \Sigma_y \to \mathbb{R}^{d_e}$ where embedding size is $d_e$  


**Encoder**  
BiLSTM maps input sequence $\mathbf{x}$ to concatenation of final hidden states ${\mathbf{h}}_{|\mathbf{x}|}^{(enc)} = \overrightarrow{\mathbf{h}}_{|\mathbf{x}|}^{(enc)} \oplus \  \overleftarrow{\mathbf{h}}_{|\mathbf{x}|}^{(enc)}$  
    - Hidden size is $d_h \in \{100, 200, 400, 600\}$  
    - Initial hidden state is $\mathbf{0}$ vector  
    - Intermediate hidden states are also stored (for alignment in decoding, see below)  

**Decoder**  
Unidirectional (left-to-right) LSTM maps previous (target or predicted?) output $y_{i-1}$ and hidden state $\mathbf{h}_{i-1}^{(dec)}$ to current hidden state $\mathbf{h}_i^{(dec)}$ of size $d_h \in \{100, 200, 400, 600\}$

**Alignment**  
Softmax alignment distribution over alignments to single symbols in input at step $i$:  

$\alpha_j(i) = {\Large \frac{\exp(e_{ij})}{\sum_{j'=1}^{|\mathbf{x}|} \exp(e_{ij'})}}$

where $\alpha_j(i)$ is the probability of aligning output $y_i$ to input $x_j$  

$e_{ij} = (\mathbf{h}_i^{(dec)})^T \ \mathbf{T} \ \mathbf{h}_j^{(enc)}$

**Output likelihoods**

Probability of output $y_i$ conditioned on alignment $a_i \in \{1, \ldots, |\mathbf{x}|\}$

$p(y_i | a_i, \mathbf{y}_{< i}, \mathbf{x}) = \texttt{softmax}\left(\mathbf{W} \ \mathbf{f}(\mathbf{h}_i^{(dec)}, \mathbf{h}^{(enc)}_{a_i}) \right)$  

$\mathbf{f}(\mathbf{h}_i^{(dec)}, \mathbf{h}^{(enc)}_{a_i}) = \texttt{tanh}(\mathbf{S} \ (\mathbf{h}_i^{(dec)} \oplus \mathbf{h}_{a_i}^{(enc)}))$

where $\mathbf{S}$ is size $ \mathbb{R}^{d_s \times 3d_h}$ and here $d_s = 3d_h$

Probability of output $y_i$ marginalized over alignments ("mixture of softmaxes")

$p(y_i | \mathbf{y}_{< i}, \mathbf{x}) = \sum_{a_i = 1}^{|\mathbf{x}|} p(y_{i} | a_i, \mathbf{y}_{< i}, \mathbf{x}) \ p(a_i | \mathbf{y}_{< i}, \mathbf{x})$  
$\qquad = \sum_{j = 1}^{|\mathbf{x}|} p(y_i | a_{ij}, \mathbf{y}_{< i}, \mathbf{x}) \ \alpha_i(j)$

**Generation**

"Because we did not observe any improvements in preliminary experiments when decoding with beam search, all models are decoded greedily."

**Training**

"We train the model with Adam ... with an initial learning rate of $0.001$. We halve the learning rate whenever the development log-likelihood doesn't improve. We stop after the learning rate dips to $1 \times 10^{-5}$. We save all models after each epoch and select the model with the best development performance. We train the model for at most $50$ epochs, though all the experiments stop early. We train ... with batch sizes of [20 or 50]. ... We apply gradient clipping to the large model with maximum gradient norm 5."

**Parameter count**

1.199M (small), 8.621 M (large)