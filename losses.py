import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

# DCGAN loss
class Loss_obj(object):
  def __init__(self, which_train_fn = 'AE', is_sharpen = 0.0,
               dim_z = 128, is_dec_var= False, add_data_noise = False, dataset_channel = True,
               l_p_z = 1, l_q_z_x = 1, l_p_z_c = 1, l_p_c = 1, l_q_y_z = 1,
               mult_par = 0, largest_seq = 200, is_double_loss = False, 
               dim_f = 0, use_extra_feat = False, is_anomaly_detection = False, n_classes = 10,
               num_neural_classifier = False, noise_data = False,
               beta_1 = 0.0, beta_2 = 0.0, **kwargs):

    self.dim_z           = dim_z
    self.dim_f           = dim_f
    self.is_sharpen      = is_sharpen
    self.sharpen         = dim_z
    if use_extra_feat:
      self.sharpen = dim_z  + dim_f
    self.is_dec_var      = is_dec_var
    self.mult_par        = mult_par
    self.largest_seq     = largest_seq
    self.dataset_channel = dataset_channel 
    self.which_train_fn  = which_train_fn
    self.is_anomaly_detection = is_anomaly_detection
    self.n_classes       = n_classes

    self.beta_1          = beta_1
    self.beta_2          = beta_2

    self.cross_entropy = nn.CrossEntropyLoss()
    self.log_likelihood   = self.neg_MSE
    self.num_neural_classifier = num_neural_classifier

    if self.is_dec_var and noise_data:
        self.log_likelihood = self.gaussian_log_gaussian
        #self.log_likelihood = self.gaussianlv0_log_gaussian
    elif self.is_dec_var:
        self.log_likelihood = self.gaussianlv0_log_gaussian

    if self.which_train_fn == 'VAE':
        self.obtain_reg_loss = self.VAE_reg
    if self.which_train_fn == 'VADE':
        self.obtain_reg_loss = self.VADE_reg

    self.l_p_z   = l_p_z
    self.l_p_z_c = l_p_z_c
    self.l_p_c   = l_p_c
    self.l_q_z_x = l_q_z_x
    self.l_q_y_z = l_q_y_z
    
    self.fdim = [1, 2]
    self.mask_mean_sum = self.mask_mean_sum_2f

  def cross_entropy_st(self, labels, log_y_pred):
    return - (log_y_pred[torch.arange(len(labels)), labels]).mean(0)

  def cross_entropy_mt(self, labels, log_y_pred, mask_used):
    #log_y_pred: Batch x Time x labels
    bs, time, n_classes = log_y_pred.shape
    p_logits = log_y_pred.reshape(-1, n_classes)[torch.arange(bs * time), labels.reshape(-1, 1).repeat(1, time).reshape(-1,)]
    p_logits = p_logits.reshape(bs, time)
    cross_entropy = torch.mean(torch.sum(p_logits * mask_used.squeeze(), -1) / mask_used.sum(-1), 0)
    #cross_entropy = -(log_y_pred[torch.arange(len(labels)), torch.arange(log_y_pred.sshape[1]), labels])
    # cross_entropy = torch.mean(torch.sum(cross_entropy * mask_used.squeeze(), -1) / mask.sum(-1), 0)
    return cross_entropy

  def mean_sum(self, loss_func, mask, not_batch_mean = False):
    if mask is None:
      if not not_batch_mean:
        return torch.sum(loss_func , self.fdim).mean()
      else:
        return torch.sum(loss_func , self.fdim)
    else:
      return self.mask_mean_sum(loss_func, mask, not_batch_mean = not_batch_mean)

  def mask_mean_sum_2f(self, loss_func, mask, not_batch_mean = False):
      loss_func = loss_func * mask
      if not not_batch_mean:
        return torch.mean(torch.sum( torch.sum(loss_func, 1) / (mask.sum(1)  + 1e-12) , 1 ) )
      else:
        return torch.sum( torch.sum(loss_func, 1) / (mask.sum(1)  + 1e-12) , 1 ) 

  def neg_MSE(self, x_mu,  default, x_mu_pred, default2, mask = None, not_batch_mean = False):
    loss_func = - (x_mu - x_mu_pred).pow(2)
    return self.mean_sum(loss_func, mask, not_batch_mean = not_batch_mean)

  def neg_WMSE(self, x_mu, x_var, x_mu_pred, default2, mask = None, not_batch_mean = False):
    loss_func = - ((x_mu - x_mu_pred).pow(2))/x_var
    return self.mean_sum(loss_func, mask, not_batch_mean = not_batch_mean)

  def gaussianlv0_log_gaussian(self, x_mu, default, x_mu_pred, x_var_pred, mask = None, not_batch_mean = False):
    loss_func = - 0.5 * ((x_mu - x_mu_pred).pow(2) / (x_var_pred + 1e-8) + x_var_pred.log() + np.log(2 * math.pi) )
    return self.mean_sum(loss_func, mask, not_batch_mean = not_batch_mean)

  def gaussian_log_gaussian(self, x_mu, x_var, x_mu_pred, x_var_pred, mask = None, not_batch_mean = False):
    x_var = x_var.clip(0,3)
    loss_func = - 0.5 * (np.log(2*math.pi) + (x_var_pred + 1e-8).log() +  x_var/ (x_var_pred  +  1e-8) + (x_mu - x_mu_pred).pow(2)/ (x_var_pred + 1e-8))
    return self.mean_sum(loss_func, mask, not_batch_mean = not_batch_mean)

  def log_gaussian(self, lv):
    aux = - 0.5 * (lv + np.log(2*math.pi) + 1)
    return aux.sum(dim=- 1)

  def log_gaussian_prior(self, z_mu, lv):
    aux = - 0.5 * (z_mu**2 + lv.exp() + np.log(2*math.pi) )
    return aux.sum(dim=- 1)

  def rec_loss_interp_pred(self, x, x_pred, mask, mask_drop):
    """ Autoencoder loss """
    mask = mask * (1 - mask_drop)
    return - self.neg_MSE(x, None, x_pred, None, mask)

  def compute_params(self, z, gamma):
    N = gamma.size(0)
    sum_gamma = torch.sum(gamma, dim=0)
    phi = sum_gamma / N
    mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
    # z = N x D
    # mu = K x D
    # gamma N x K
    # z_mu = N x K x D
    z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
    # z_mu_outer = N x K x D x D
    z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
    # K x D x D
    cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
    return phi, mu, cov

  def compute_energy(self, z, phi, mu, cov, obtain_reg = False):
    k, D, _ = cov.size()
    z_mu    = z.unsqueeze(1) - mu.unsqueeze(0)

    eps = 1e-3
    torch_eye   = torch.eye(D, device=cov.device)
    diag_noise  = torch_eye.unsqueeze(0) * eps 
    cov = cov + diag_noise

    cov_inv     = torch.inverse(cov)
    cov_logdet  = torch.logdet(cov)

    # N x K
    exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inv.unsqueeze(0), dim=-2) * z_mu, dim=-1)
    # for stability (logsumexp)
    max_val  = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]
    exp_term = torch.exp(exp_term_tmp - max_val)
    cte_term = (0.5*(cov_logdet + D * np.log(2 * np.pi))    ).exp()

    sample_energy = - (max_val.squeeze() + (phi.unsqueeze(0) * exp_term / (cte_term + eps)  + eps).sum(-1).log()  )

    if not obtain_reg:
        return sample_energy
    else:
        return (sample_energy, (1/torch.diagonal(cov, dim1= -2, dim2= -1)).sum()  )

  def distances(self, x, x_pred, mask):
    x_norm              = (x.pow(2) * mask ).sum(2)
    x_pred_norm         = (x_pred.pow(2) * mask ).sum(2)
    euclidean_distance  = ((x_pred - x).pow(2) * mask).sum(2) / x_norm
    cosine_distance     = (x * x_pred * mask ).sum(2) / x_norm / x_pred_norm
    return euclidean_distance.mean(-1).unsqueeze(1), cosine_distance.mean(-1).unsqueeze(1)

  def compute_params_and_energy(self, x, x_pred, mask, labels, z, classifier, extra_feat = None, 
                                                      train = False, obtain_just_params = False, prior_params = None):
    euc, cos      = self.distances(x, x_pred, mask)
    z_rep         = z
    if extra_feat is not None:
      zc            = torch.cat((z_rep, euc, cos, extra_feat), dim=1)
    else:
      zc            = torch.cat((z_rep, euc, cos), dim=1)
    if train:
      log_y_pred    = classifier.log_latent_classifier(zc, feat = None, obtain_embedding = False)
    else:
      log_y_pred, z_classifier    = classifier.log_latent_classifier(zc, feat = None, obtain_embedding = True)

    if prior_params is None:
      phi, mu, cov  = self.compute_params(zc, log_y_pred.exp() )
    else:
      phi, mu, cov = prior_params

    if obtain_just_params:
      return log_y_pred.exp(), mu, cov

    energy_loss   = self.compute_energy(zc, phi, mu, cov, obtain_reg = self.beta_2 > 0 and train)

    reg_loss = 0
    if self.beta_2 > 0 and train:
      energy_loss, reg_loss = energy_loss
    ce_loss  = 0
    if train:
      ce_loss       = - (log_y_pred[torch.arange(len(labels)), labels]).sum(-1).mean(0)
    acc             =   (labels  == log_y_pred.argmax(1)).float().mean().item()

    if not train:
      return ce_loss, energy_loss, reg_loss, log_y_pred, acc, z_classifier
    else:
      return ce_loss, energy_loss, reg_loss, log_y_pred, acc

  def VAE_reg(self, z_mu, z_lv, obtain_dict = False):
    E_q_z_x = self.log_gaussian(z_lv).mean(0)
    E_p_z   = self.log_gaussian_prior(z_mu, z_lv).mean(0)

    l_p_z   = self.l_p_z
    l_q_z_x = self.l_q_z_x

    if not obtain_dict:
      return (l_p_z * E_p_z - l_q_z_x* E_q_z_x)
    else:
      reg_dict = {}

      reg_dict['E_q_z_x'] = E_q_z_x.item()
      reg_dict['E_p_z']   = E_p_z.item()
      return  (l_p_z * E_p_z - l_q_z_x* E_q_z_x), reg_dict

  def gmm_membressy(self, z_samples, mu_c, lv_c):
    c_mu   = mu_c.unsqueeze(0)
    c_var  = lv_c.unsqueeze(0).exp()
    z = z_samples.unsqueeze(1).repeat(1, c_mu.size()[0], 1)
    log_dist = (- 0.5 * (2 * math.pi * c_var).log() - (z - c_mu).pow(2) / (2 * c_var)).sum(2)
    if self.is_sharpen:
      log_dist = log_dist/self.is_sharpen
    gamma = log_dist.exp()
    den = gamma.sum(1).unsqueeze(1) + 1e-20
    pred = gamma / den
    return pred

  def gaussian_log_gaussian_c(self, z_mu, z_logvar, mu_c, lv_c, feat = None):  ### z_mu = [batch, dim]
    ### z_mu = [batch, dim]
    ### c_mu = [n_clusters, dim]
    ### result = [batch, n_clusters]
    pi = torch.tensor(math.pi)
    if feat is not None:
      c_mu_feat      = mu_c[:, self.dim_z:].unsqueeze(0)
      c_var_feat     = lv_c[:, self.dim_z:].unsqueeze(0).exp()
      feat_reap      = feat.unsqueeze(1).repeat(1, mu_c.size()[0], 1)
      dist_feat      = torch.sum(- 0.5 * ((2 * pi).log() + c_var_feat.log() + \
                                           (feat_reap - c_mu_feat).pow(2) / c_var_feat), -1)
      mu_c           = mu_c[:, :self.dim_z]
      lv_c           = lv_c[:, :self.dim_z]

    z_mu       = z_mu.unsqueeze(1).repeat(1, mu_c.size()[0], 1)
    z_logvar   = z_logvar.unsqueeze(1).repeat(1, mu_c.size()[0], 1)
    c_mu       = mu_c.unsqueeze(0)
    c_var      = lv_c.unsqueeze(0).exp()

    dist = torch.sum(- 0.5 * ((2 * pi).log() + c_var.log() + z_logvar.exp() / c_var + (z_mu - c_mu).pow(2) / c_var), -1)
    if feat is not None:
      dist += dist_feat   
    return dist

  def gaussian_log_gaussian_c_mm(self, z_mu, z_logvar, mu_c, lv_c, feat = None):  ### z_mu = [batch, dim]
    ### z_mu = [batch, dim]
    ### c_mu = [n_clusters, dim]
    ### result = [batch, n_clusters]
    pi = torch.tensor(math.pi)
    if feat is not None:
      c_mu_feat      = mu_c[:, :, self.dim_z:].unsqueeze(0)
      c_var_feat     = lv_c[:, :, self.dim_z:].unsqueeze(0).exp()
      feat_reap      = feat.unsqueeze(1).unsqueeze(2).repeat(1, mu_c.size()[0], mu_c.size()[1], 1)
      dist_feat      = torch.sum(- 0.5 * ((2 * pi).log() + c_var_feat.log() + \
                                           (feat_reap - c_mu_feat).pow(2) / c_var_feat), -1)
      mu_c           = mu_c[:, :, :self.dim_z]
      lv_c           = lv_c[:, :, :self.dim_z]
      
    z_mu       = z_mu.unsqueeze(1).unsqueeze(2).repeat(1, mu_c.size()[0], mu_c.size()[1], 1)
    z_logvar   = z_logvar.unsqueeze(1).unsqueeze(2).repeat(1, mu_c.size()[0], mu_c.size()[1], 1)
    c_mu       = mu_c.unsqueeze(0)
    c_var      = lv_c.unsqueeze(0).exp()

    dist = torch.sum(- 0.5 * ((2 * pi).log() + c_var.log() +
                      z_logvar.exp() / c_var + (z_mu - c_mu).pow(2) / c_var), -1)
    if feat is not None:
      dist += dist_feat   
 
    return dist

  def mi_zy_x(self, z_mu, z_lv, GMM_mu, GMM_lv, feat = None, Prior = None, obt_extended_score = False):
    E_q_z_x = self.log_gaussian(z_lv)
    if not obt_extended_score:
      E_p_z_c = self.E_p_z_c(z_mu, z_lv, GMM_mu, GMM_lv, feat, Prior)
      return E_p_z_c - E_q_z_x
    else:
      E_p_z_c, E_p_z_c_not_summed, glog_g_c = self.E_p_z_c(z_mu, z_lv, GMM_mu, GMM_lv, feat,
                                                     Prior, obt_extended_score = obt_extended_score)
      return E_p_z_c - E_q_z_x, E_p_z_c_not_summed - E_q_z_x.unsqueeze(1), glog_g_c

  def E_p_z_c(self, z_mu, z_lv, GMM_mu, GMM_lv, feat = None, Prior = None, obtain_y_pred = False,
                                                               z = None, obt_extended_score = False):

    if not self.num_neural_classifier:
      log_y_pred = self.log_gmm_membressy(z_mu if z is None else z, GMM_mu, GMM_lv, feat = feat)
    else:
      log_y_pred = Prior.log_latent_classifier(z_mu if z is None else z, feat = feat)
    y_pred     = log_y_pred.exp()

    gaussian_log_gaussian_c = self.gaussian_log_gaussian_c(z_mu, z_lv, GMM_mu, GMM_lv, feat = feat)
    
    E_p_z_c = (y_pred * gaussian_log_gaussian_c)
    if not obtain_y_pred:
      if not obt_extended_score:  
        return E_p_z_c.sum(1)
      else:
        return E_p_z_c.sum(1), E_p_z_c, gaussian_log_gaussian_c
    else:
      return E_p_z_c.sum(1), y_pred, log_y_pred

  def log_gmm_membressy(self, z_samples, mu_c, lv_c, feat = None): ### z_mu = [batch, dim]
    ### z_mu = [batch, dim]
    ### c_mu = [n_clusters. dim]
    ### result = [batch, n_clusters]

    if feat is not None:
      c_mu_feat      = mu_c[:, self.dim_z:].unsqueeze(0)
      c_var_feat     = lv_c[:, self.dim_z:].unsqueeze(0).exp()
      feat_reap      = feat.unsqueeze(1).repeat(1, mu_c.size()[0], 1)
      exp_feat_part  = ((feat_reap - c_mu_feat).pow(2) / (2 * c_var_feat)).sum(2)
      dist_feat      = (- 0.5 * (2 * math.pi * c_var_feat).log()).sum(2) - exp_feat_part

    mu_c           = mu_c[:, :self.dim_z]
    lv_c           = lv_c[:, :self.dim_z]

    c_mu            = mu_c.unsqueeze(0)
    c_var           = lv_c.unsqueeze(0).exp()
    z               = z_samples.unsqueeze(1).repeat(1, mu_c.size()[0], 1)
    exp_part        = ((z - c_mu).pow(2) / (2 * c_var)).sum(2)
    dist            = (- 0.5 * (2 * math.pi * c_var).log()).sum(2) - exp_part

    if feat is not None:
      dist = dist + dist_feat
    if self.is_sharpen:
      dist = dist/self.is_sharpen
    closer            = dist.max(1)[0].unsqueeze(1)
    log_pred_final    = dist - closer - torch.log ((dist - closer).exp().sum(1)).unsqueeze(1)
    return log_pred_final

  def log_gmm_membressy_mm(self, z_samples, mu_c, lv_c, feat = None): ### z_mu = [batch, dim]
    ### z_mu = [batch, dim]
    ### c_mu = [n_clusters, n_modes, dim]
    ### result = [batch, n_clusters]

    if feat is not None:
      c_mu_feat      = mu_c[:, :, self.dim_z:].unsqueeze(0)
      c_var_feat     = lv_c[:, :, self.dim_z:].unsqueeze(0).exp()
      feat_reap      = feat.unsqueeze(1).unsqueeze(2).repeat(1, mu_c.size()[0], mu_c.size()[1], 1)
      exp_feat_part  = ((feat_reap - c_mu_feat).pow(2) / (2 * c_var_feat)).sum(3)
      dist_feat      = (- 0.5 * (2 * math.pi * c_var_feat).log()).sum(3) - exp_feat_part

    mu_c           = mu_c[:, :, :self.dim_z]
    lv_c           = lv_c[:, :, :self.dim_z]

    c_mu            = mu_c.unsqueeze(0)
    c_var           = lv_c.unsqueeze(0).exp()
    z               = z_samples.unsqueeze(1).unsqueeze(2).repeat(1, mu_c.size()[0], mu_c.size()[1], 1)
    exp_part        = ((z - c_mu).pow(2) / (2 * c_var)).sum(3)
    dist            = (- 0.5 * (2 * math.pi * c_var).log()).sum(3) - exp_part

    if feat is not None:
      dist = dist + dist_feat
    if self.is_sharpen:
      dist = dist/self.is_sharpen
    closer            = dist.max(2)[0].unsqueeze(2)
    log_pred_final    = dist - closer - torch.log ((dist - closer).exp().sum(2)).unsqueeze(2)
    return log_pred_final

  def VADE_reg(self, z, z_mu, z_lv, GMM_mu, GMM_lv, GMM_phi, y = None, log_y_pred = None, feat = None, Prior = None):

      log_lik_reg = 0

      if log_y_pred is None:
        E_p_z_c, y_pred, log_y_pred = self.E_p_z_c(z_mu, z_lv, GMM_mu, GMM_lv,  feat, Prior, obtain_y_pred = True, z = z)
      else:
        y_pred  = log_y_pred.exp()
        E_p_z_c = self.E_p_z_c(z_mu, z_lv, GMM_mu, GMM_lv, feat, Prior, obtain_y_pred = False, z = z)

      E_p_z_c = E_p_z_c.mean(0)
      # E_p_c is already marginalized

      E_p_c   = (y_pred.mean(0) * torch.log(self.prior_y)).sum()
      E_q_z_x = self.log_gaussian(z_lv).mean(0)
      E_q_y_z = (y_pred * log_y_pred).sum(1).mean(0)

      l_p_z_c = self.l_p_z_c
      l_p_c   = self.l_p_c
      l_q_z_x = self.l_q_z_x
      l_q_y_z = self.l_q_y_z

      log_lik_reg += (l_p_z_c * E_p_z_c - l_q_z_x * E_q_z_x - l_q_y_z * E_q_y_z).mean(0) + l_p_c * E_p_c

      ### self.VAE_reg(z_par_mu, z_par_lv) Regularization for kernel parameters
      cross_entropy = None
      if y is not None:
        acc           =  (y  == y_pred.argmax(1)).float().mean()

      reg_dict = {}
      if y is not None:
        reg_dict['E_acc_train']       = acc.item()
      reg_dict['E_p_z_c']         = l_p_z_c * E_p_z_c.item()
      reg_dict['E_p_c']           = l_p_c   * E_p_c.item()
      reg_dict['E_q_z_x']         = l_q_z_x * E_q_z_x.item()
      reg_dict['E_q_y_z']         = l_q_y_z * E_q_y_z.item()

      return log_lik_reg, cross_entropy, reg_dict