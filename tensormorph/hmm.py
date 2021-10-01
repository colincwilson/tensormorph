# -*- coding: utf-8 -*-

import config
from tpr import *


class HMM(nn.Module):
    """
    HMM allowing insertions and deletions (skips), 
    similar to Profile HMM of bioinformatics except  
    with tied insertion/deletion probabilities
    todo: interpret word boundary specification in decoder output 
    (not target) as gradient transition to unique final state
    """

    def __init__(self, n):
        super(HMM, self).__init__()
        #n = config.nrole # xxx allow extra decode steps?
        nstate = n * 3 + 1  # last state is begin
        # Move logits (w_match, w_insert, w_delete)
        w = torch.nn.Parameter(0.1 * torch.randn(3))

        # Regular transitions into match and insert states
        # (convention: 0 indicates no transition)
        Tmatch = torch.zeros(n, nstate, requires_grad=False)
        Tinsert = torch.zeros(n, nstate, requires_grad=False)
        for i in range(n - 1):
            Tmatch[i + 1, i] = 1.0  # match[i] -> match[i+1]
            Tmatch[i + 1, i + n] = 1.0  # insert[i] -> match[i+1]
            Tmatch[i + 1, i + 2 * n] = 1.0  # delete[i] -> match[i+1]
            Tinsert[i, i] = 1.0  # match[i] -> insert[i]
            Tinsert[i, i + n] = 1.0  # insert[i] -> insert[i]
            Tinsert[i, i + 2 * n] = 1.0  # delete[i] -> insert[i]

        # Regular transitions into delete states
        Tdelete = torch.zeros(n, nstate, requires_grad=False)
        for i in range(n):
            for j in range(i + 1, n):
                Tdelete[j, i] = float(j - i)  # match[i] -> delete[j]
                Tdelete[j, i + n] = float(j - i)  # insert[i] -> delete[j]

        # Initial transitions, from begin state
        Tmatch[0, -1] = Tinsert[0, -1] = 1.0
        Tdelete[:, -1] = torch.arange(1, n + 1, dtype=torch.float)

        self.n = n
        self.nstate = nstate
        self.w = w
        self.Tmatch = Tmatch
        self.Tinsert = Tinsert
        self.Tdelete = Tdelete

    def forward(self, Y, targ, m, ignore_index=None):
        """
        Forward algorithm for HMMs
        """
        nbatch = 2  # xxx get batch from X or targ
        n = self.n
        nstate = self.nstate
        logeps_ = np.log(1.0e-7)  # np.log(0.0)
        logeps = logeps_ * torch.ones(nbatch, 1)

        # Move log probs
        #w = log(torch.softmax(self.w, dim=0))
        w = log(torch.tensor([0.8, 0.1, 0.1]))
        w_match, w_insert, w_delete = w.unbind(0)

        # Transition log probs
        # (premultiply by move weights, nullify
        # entries for absent transitions)
        Tmatch = w_match * self.Tmatch
        Tinsert = w_insert * self.Tinsert
        Tdelete = w_delete * self.Tdelete
        Tmatch.masked_fill_(self.Tmatch == 0.0, logeps_)
        Tinsert.masked_fill_(self.Tinsert == 0.0, logeps_)
        Tdelete.masked_fill_(self.Tdelete == 0.0, logeps_)

        # Normalize transition log probs by source state
        T = torch.cat([Tmatch, Tinsert, Tdelete], 0)
        T = T - logsumexp(T, 0, keepdim=True)
        T = T.unsqueeze(0)
        Tmatch, Tinsert, Tdelete = T.chunk(3, dim=1)

        # Initialize forward log probs
        alpha = torch.zeros(nbatch, nstate, requires_grad=False)
        alpha.fill_(logeps_)
        alpha[:, -1] = 0.0  # start in begin state
        #print(alpha)

        # Emission log probs
        k = 10  # number of symbols
        Y = log(1 / float(k) * torch.ones(nbatch, k, n))  # xxx
        I = log(1 / float(k) * torch.ones(nbatch, k))  # xxx
        I = I.unsqueeze(-1).expand(nbatch, k, n)

        # xxx targets
        targ = torch.ones(nbatch, n, dtype=torch.long)
        targ[:, 4] = 0

        # Recursion
        for t in range(n):
            # Target at output position t
            targ_t = targ[:, t].unsqueeze(-1).unsqueeze(-1)
            targ_t = targ_t.expand(nbatch, 1, n)  # Broadcast over input posns

            # Split alpha for separate updates
            alpha_match, alpha_insert, alpha_delete, alpha_begin = \
                alpha.split(n, dim = 1)

            # Update alpha for deletion states
            # alpha[b,j] = sum_k T[k->j] * alpha[b,k]
            trans = Tdelete + alpha.unsqueeze(1)  # Broadcast over dest
            alpha_delete = logsumexp(trans, dim=-1)  # Sum over src
            alpha = torch.cat(
                [alpha_match, alpha_insert, alpha_delete, alpha_begin], -1)

            # Update alpha for match states
            # alpha[b,i] = {sum_k T[j->i] * alpha[b,j]} * emit[b,i]
            # where emit[b,i] = prob(targ[b,t] | match[b,i])
            trans = Tmatch + alpha.unsqueeze(1)
            trans = logsumexp(trans, dim=-1)
            emit = torch.gather(Y, 1, targ_t)
            if ignore_index is not None:  # Treat ignore_index as wildcard
                mask = (targ_t != ignore_index).float()
                mask = log(mask)
                emit = emit * mask
            alpha_match = trans + emit.squeeze(1)

            # Update insert state log probs
            # alpha[b,i] = {sum_k T[j->i] * alpha[b,j]} * emit[b,i]
            # where emit[b,i] = prob(targ[b,t] | insert[b,i])
            trans = Tinsert + alpha.unsqueeze(1)
            trans = logsumexp(trans, dim=-1)
            emit = torch.gather(I, 1, targ_t)
            if ignore_index is not None:  # Treat ignore_index as wildcard
                mask = (targ_t != ignore_index).float()
                emit = emit * mask
            alpha_insert = trans + emit.squeeze(1)

            # Combine alpha updates
            alpha = torch.cat([alpha_match, alpha_insert, alpha_delete, logeps],
                              -1)

        # Sum over final match, insert, delete states
        fin = torch.stack(
            [alpha_match[:, -1], alpha_insert[:, -1], alpha_delete[:, -1]], -1)
        fin = -logsumexp(fin, -1)
        #print(fin)
        return fin


def main():
    hmm = HMM(5)
    fin = hmm.forward(None, None, 0, ignore_index=-1)
    print(fin)


if __name__ == "__main__":
    main()