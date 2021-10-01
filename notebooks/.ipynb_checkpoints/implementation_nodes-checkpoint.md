
7/09/2020

- **Notation**  
    ε &nbsp; empty (all-zero vector)  
    ⋊ &nbsp; beginning-of-string delimiter (also >)  
    ⋉ &nbsp; end-of-string delimiter (also <)  
    □ &nbsp; wildcard (average of ordinary symbol vectors)  

- **Pivots**. The model defines a set of _pivots_, locations at which an affix can be inserted in a stem.
    - The possible pivots are {at, before} the {first, last} instance of {-, ⋊, ⋉, C, V} where '-' indicates no pivoting.  
    - Choice among the pivot points is parameterized by **logits**. There are two logit parameters for each of the binary choices {at, before} and {first, last} and five logit parameters for {-, ⋊, ⋉, C, V}.  
    - Logits $\mathbf{\alpha}$ can be mapped to gradient and discrete choices in the following ways:

        $$p(k) = \frac{\exp(\alpha_k/T)}{\sum_{k'} \exp(\alpha_{k'}/T)} \quad \text{// deterministic softmax}$$

        $$p(k) = \textrm{gumbel_softmax}(\alpha/T)_k \quad \text{// stochastic softmax}$$

        where $T > 0$ is the temperature and the output is optionally discretized to a one-hot vector (by setting the maximal component of $p$ equal to $1$ and all other components equal to $0$). For fixed input $\mathbf{\alpha}$, $\textrm{softmax}$ with or without discretization is deterministic, while the output of $\textrm{gumbel_softmax}$  varies by random sampling.

- **Truncation**. The model uses pivots to define a number of possible stem _truncations_.
    - A stem can be truncated {before, after} a pivot. Separately, the begin and/or end delimiter of a stem can be truncated (this is needed for reduplication).  
    - Choice among truncations is parameterized by logits: the parameters for the truncation pivot (as above), two logits for {before, after}, and two logits each for the begin and end delimiter.  
    - Truncation parameters can be mapped to gradient or discrete choices as above.

Ignoring the form of the affix, which has $\sum_{n=1}^{n_{max}} |\Sigma|^n$ possible discrete values, the pivot and truncation parameters alone define a space of 16,000 possible discrete affixation patterns.

---

- Local minima. In learning infixation such as (>kida<, >kumida<) with gradient descent, the model often enters a local minumum similar to:

stem: >● k i d a <  
affix: □ u m●  
output: > k/□ u m i d a <

where '●' marks the pivot in the stem and the end of the affix. The correct analysis is instead:

stem: > k● i d a <  
affix: u m●  
output: > k u m i d a <

Escaping from the local minimum to the correct analysis requires simultaneously changing (i) the pivot (from after > to after the first C or before the first V), (ii) the form of the affix (shifting □um to um), and (iii) the endpoint of the affix (from the second to the third position). Making one change, such as:

stem: > k● i d a <  
affix: □ u m●  
output: > k □ u m i d a <

results in an output that is incorrect and misaligned with the target after all but the first two symbols.

