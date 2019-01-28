import torch
import torch.nn as nn

from src import model_modules

class Classifier(nn.Module):
    def __init__(self, in_c=3, z_dim=512, h_dim=128, n_act=10, n_id=9):
        super().__init__()
        self.in_c = in_c
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.n_act = n_act
        self.n_id = n_id

        self.enc = model_modules.Encoder(in_c=in_c, z_dim=z_dim)
        self.attr_net = model_modules.AttributeNet(z_dim=z_dim, h_dim=128, n_act=n_act, n_id=n_id)

    def forward(self, x):
        x_enc = self.enc(x)
        out_act, out_id = self.attr_net(x_enc)
        return out_act, out_id

class VideoVAE(nn.Module):
    def __init__(self, z_dim=512, h_dim=512, n_act=10, n_id=9, input_size=512*3, hidden_size=512, num_layers=1, bidirectional=False):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.n_act = n_act
        self.n_id = n_id

        # Encoder and attr_net can be trained indepently
        self.enc = model_modules.Encoder()
        self.attr_net = model_modules.AttributeNet(z_dim=z_dim, h_dim=128, n_act=n_act, n_id=n_id)

        # Conditional Approximate Posterior
        self.post_q = model_modules.DistributionNet(in_dim=z_dim,
                                      h_dim=128, 
                                      out_dim=512)
        self.post_a = model_modules.DistributionNet(in_dim=(z_dim*2+n_act+n_id),
                                      h_dim=128, 
                                      out_dim=512)
        self.post_dy = model_modules.DistributionNet(in_dim=(z_dim+z_dim+z_dim),
                                       h_dim=128,
                                       out_dim=512)

        self.mlp_lstm = model_modules.MLP(in_dim=512, h_dim=128, out_dim=512)

        # Prior
        self.prior = model_modules.DistributionNet(in_dim=(z_dim+n_act+n_id),
                                     h_dim=128,
                                     out_dim=512)

        # Decoder
        self.dec = model_modules.Decoder()

        # LSTM
        self.input_size = input_size # dimension: (z_dim+z_dim*2),  z_t and [mu_q, logvar_q]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # inputs:  input, (h_0, c_0)
        # outputs: output, (h_n, c_n)
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            bidirectional=False)

        if self.bidirectional:
            self.lstm_backward = nn.LSTM(input_size=input_size, 
                                         hidden_size=hidden_size, 
                                         num_layers=num_layers, 
                                         bidirectional=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, val=0)

    def pred_attr(self, x):
        """Classify the action and identity."""
        x_enc = self.enc(x)
        logit_act, logit_id = self.attr_net(x_enc)
        _, pred_act = logit_act.max(1)
        _, pred_id = logit_id.max(1)
        return pred_act, pred_id

    def seq_cls(self, img_seq):
        """ Classify the whole sequence to get the attributes. """
        b, t, c, h, w = img_seq.shape
        img_seq_expand = img_seq.view(-1, c, h, w)
        pred_act, pred_id = self.pred_attr(img_seq_expand)

        # view back to b, t
        pred_act, pred_id = pred_act.view(b, t), pred_id.view(b, t)

        # voting
        pred_act_mode, _ = torch.mode(pred_act, dim=1, keepdim=True)
        pred_id_mode, _  = torch.mode(pred_id, dim=1, keepdim=True)

        # repeat
        pred_act_mode = pred_act_mode.repeat(1, 10)
        pred_id_mode = pred_id_mode.repeat(1, 10)

        # vutils.save_image(img_seq_expand*.5 + .5, 'hey.jpg', nrow=10)
        return pred_act, pred_id

    def cond_approx_posterior(self, x_enc, a_label, phi_h):
        # For Posterior q
        mu_q, logvar_q = self.post_q.encode(x_enc)

        # For Posterior a
        #   phi_q: [mu_q, logvar_q]
        #   merge: (phi_q, a_label)
        #   phi_q_merged: (phi_q, a_label)
        phi_q_merged = torch.cat([mu_q, logvar_q, a_label], dim=1) # NOTE: merge along z-axis
        mu_a, logvar_a = self.post_a.encode(phi_q_merged)

        # For Posterior dy
        #   phi_a: [mu_a, logvar_a, ]
        #   merge: (phi_a, \phi(h_{t-1}))
        phi_a_merged = torch.cat([mu_a, logvar_a, phi_h[0]], dim=1)
        z_dy, mu_dy, logvar_dy = self.post_dy(phi_a_merged)

        return mu_q, logvar_q, z_dy, mu_dy, logvar_dy

    def lstm_forward(self, mu_q, logvar_q, z_t, h_prev, c_prev):
        """LSTM forward for 1 time step.
        This only propagate 1 time step, so lstm_output should be the same as h_t
        
        params:
            - mu_q, logvar_q
            - z_t: samples from prior (z_p) or posterior (z_dy)
            - h_prev: hidden states from t-1
            - c_prev: cell states from t-1
        returns:
            - lstm_output: the output of LSTM. It contains all the *h_t* for eatch t. Here we have t=1, thus it should be equal to h_t
            - h_t: contains the *last* hiddent states for each time step. (n_layers*)
            - c_t: cell states for t
        notes:
            - lstm_input: merged of (z_t, mu_q, logvar_q).
                          Has to be shape: (seq_len=1, batch_size, inputdim=z_dim+z_dim*2)
        """

        lstm_input = torch.cat([z_t, mu_q, logvar_q], dim=1)
        lstm_input = lstm_input.unsqueeze(dim=0)
        lstm_output, (h_t, c_t) = self.lstm(lstm_input, (h_prev, c_prev))
        # assert (z_t - h_t).sum() == 0
        return lstm_output, h_t, c_t

    def forward(self, x_t, pred_act, pred_id, h_prev, c_prev):
        """Forward pass for the VideoVAE.
        
        The model first encode x_t into structured latent space, then sample z_t from either the 
        prior distribution (\phi_p at test time) or posterior distribution (\phi_dy at training time)
        to reconstruct x_t, with a LSTM to model the temporal information.

        For the details, please see Fig. 2 at page 5:
        https://arxiv.org/pdf/1803.08085.pdf
        """

        batch_size = x_t.size(0)

        # NOTE: no_grad here
        with torch.no_grad():
            x_enc = self.enc(x_t)

        # to one-hot
        a_label = torch.zeros(batch_size, self.n_act + self.n_id).to(x_enc)
        a_label[torch.arange(batch_size), pred_act] = 1
        a_label[torch.arange(batch_size), pred_id+self.n_act] = 1 # id: 0 -> 0 + n_act. for one-hot representation

        # transformed the h_prev
        phi_h = self.mlp_lstm(h_prev)

        # For Conditional Approximate Posterior. Check page 7 in https://arxiv.org/pdf/1803.08085.pdf
        mu_q, logvar_q, z_dy, mu_dy, logvar_dy = self.cond_approx_posterior(x_enc, a_label, phi_h)

        # For Prior p
        #   phi_h_merge: [phi_h, a]
        phi_h_merged = torch.cat([phi_h[0], a_label], dim=1)
        z_p, mu_p, logvar_p = self.prior(phi_h_merged)

        # In training, we sample from phi_dy to get z_t
        z_t = z_dy

        # LSTM forward
        lstm_output, h_t, c_t = self.lstm_forward(mu_q, logvar_q, z_t, h_prev, c_prev)

        # Reconstruction
        recon_x_t = self.dec(z_t)

        return recon_x_t, z_t, lstm_output, [h_t, c_t], [mu_p, logvar_p], [mu_dy, logvar_dy]

    def synthesize(self, x_prev, h_prev, c_prev, holistic_attr, only_prior=True, is_first_frame=True):
        """Synthesize the sequences. 
        
        Setting 1: Holistic attribute controls only. We only generate frames from prior distribution. (tend to get more blurry results.)
        Setting 2: Holistic attr. controls & first frame. The first frame is provided. Hence the generated first frame 
                   is the reconstruction of the given frame.

        For the details, please see sec. 4.2 at page 9:
        https://arxiv.org/pdf/1803.08085.pdf
        """

        holistic_act = holistic_attr['action']
        holistic_id  = holistic_attr['identity']

        batch_size = 1 # test batch_size
        ##################################
        #    synthesize for setting 1    #
        ##################################
        if only_prior:
            # transformed the h_prev
            phi_h = self.mlp_lstm(h_prev)
            a_label = torch.zeros(batch_size, self.n_act + self.n_id).to(h_prev)

            for i in range(batch_size):
                a_label[i, holistic_act] = 1
                a_label[i, self.n_act+holistic_id] = 1
            
            if is_first_frame:
                # For Prior
                phi_h_merged = torch.cat([phi_h[0], a_label], dim=1)
                z_p, mu_p, logvar_p = self.prior(phi_h_merged)

                x_gen = self.dec(z_p) * 0.5 + 0.5
                
                return x_gen, h_prev, c_prev
            else:
                # x_t: x_gen from previous step
                x_enc = self.enc(x_prev)

                # q
                mu_q, logvar_q = self.post_q.encode(x_enc)

                # p
                phi_h_merged = torch.cat([phi_h[0], a_label], dim=1)
                z_p, mu_p, logvar_p = self.prior(phi_h_merged)

                z_t = z_p

                lstm_output, h_t, c_t = self.lstm_forward(mu_q, logvar_q, z_t, h_prev, c_prev)

                # gen
                x_gen = self.dec(z_t) * 0.5 + 0.5

                return x_gen, h_t, c_t
        ##################################
        #    synthesize for setting 2    #
        ##################################
        else:
            # x_t: x_gen from previous step
            x_enc = self.enc(x_prev)

            # q
            mu_q, logvar_q = self.post_q.encode(x_enc)

            # transformed the h_prev
            phi_h = self.mlp_lstm(h_prev)

            # attr: should be control by the first frame provided by us
            # First frame in this setting is the reconstruction of the input
            a_label = torch.zeros(batch_size, self.n_act + self.n_id).to(x_enc)

            for i in range(batch_size):
                # should specify the act here
                # should not change id here
                a_label[i, holistic_act] = 1
                a_label[i, self.n_act + holistic_id] = 1

            if is_first_frame:

                mu_q, logvar_q, z_dy, mu_dy, logvar_dy = self.cond_approx_posterior(x_enc, a_label, phi_h)

                z_t = z_dy

                lstm_output, h_t, c_t = self.lstm_forward(mu_q, logvar_q, z_t, h_prev, c_prev)

                # gen
                x_gen = self.dec(z_t) * 0.5 + 0.5

            # attr: should be control by the holistic_attr provided by us
            else:
                # p
                phi_h_merged = torch.cat([phi_h[0], a_label], dim=1)
                z_p, mu_p, logvar_p = self.prior(phi_h_merged)

                z_t = z_p

                lstm_output, h_t, c_t = self.lstm_forward(mu_q, logvar_q, z_t, h_prev, c_prev)

                # gen
                x_gen = self.dec(z_t) * 0.5 + 0.5

            return x_gen, h_t, c_t

    def reset(self, batch_size=64, reset='zeros'):
        """ reset lstm state.

        Returns:
            h_0, c_0 for LSTM.
        """
        use_cuda = next(self.parameters()).is_cuda
    
        h_0 = torch.zeros(1, batch_size, self.z_dim)
        c_0 = torch.zeros(1, batch_size, self.z_dim)

        # should set to random if we are synthesizing using only prior distribution.
        if reset == 'random':
            h_0 = torch.randn_like(h_0)
            c_0 = torch.randn_like(c_0)
        
        if use_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        return h_0, c_0

    def load_cls_net(self, weight_path):
        state_dict = torch.load(weight_path)
        self.enc.load_state_dict(state_dict['encoder'])
        self.attr_net.load_state_dict(state_dict['attr_net'])