import torch
import torch.nn as nn
import torch.distributions as tdist

class VideoVAELossOld(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1Loss = nn.L1Loss(reduction='sum')


    def forward(self, recon_x_t, x_t, mu, logvar):
        L1 = self.L1Loss(recon_x_t, x_t)
        
        KLD = -.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return L1 + KLD, L1, KLD

class SimilarityLoss(nn.Module):
    def __init__(self, k=3):
        """Penalize squared error between the content feat of neighboring frames k."""
        self.k = k
        self.L2Loss = nn.MSELoss(reduction='sum')

    def forward(self, x_seq):
        loss = 0
        b, time_steps, _, _, _ = x_seq.shape
        for t in range(time_steps-self.k):
            loss += self.L2Loss(x_seq[:, t], x_seq[:, t+self.k])
        return loss


class ContentPoseAdvLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, same_seq_pair, diff_seq_pair):
        loss = None

        return loss

class VideoVAELoss(nn.Module):
    def __init__(self, recon='L1'):
        super().__init__()
        self.recon = recon
        if self.recon.lower() == 'l1':
            self.recon_loss = nn.L1Loss(reduction='sum')
        elif self.recon.lower() == 'l2':
            self.recon_loss = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError
            

    def forward(self, recon_x_t, x_t, phi_p, phi_q):
        recon_loss = self.recon_loss(recon_x_t, x_t)
        
        mu_p, logvar_p = phi_p
        mu_q, logvar_q = phi_q
        std_p, std_q = torch.exp(0.5 * logvar_p), torch.exp(0.5 * logvar_q)

        p = tdist.Normal(mu_p, std_p)
        q = tdist.Normal(mu_q, std_q)

        #KLD = torch.sum(tdist.kl.kl_divergence(p, q))
        KLD = torch.sum(tdist.kl.kl_divergence(q, p))

        return recon_loss + KLD, recon_loss, KLD