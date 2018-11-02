import torch
from torch import nn

class ColorConsistencyLoss(nn.Module):
    """
    `math`
    $$\mathcal L_{C_i} = \frac 1 n \sum ^n _{j=1}\left( \lambda_1\|\mu_{s^j_i}-\mu_{s^j_{i-1}}\|^2_2+\
    \lambda_2\|\Sigma_{s^j_i}-\Sigma_{s^j_{i-1}}\|^2_F \right)\\
    where \mu = \frac 1 N\sum_k x_k\ ,\Sigma = \frac 1 N \sum_k(x_k-\mu)(x_k-\mu)^T $$  
    """
    def __init__(self, lambda1=1, lambda2=5):
        super(ColorConsistencyLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mse_loss = nn.MSELoss()
        
    def forward(self, batch1, batch2):
        # TODO input muti_batch to reduce repeated caculation
        mu1, covariance1 = self.compute_mean_covariance(batch1)
        mu2, covariance2 = self.compute_mean_covariance(batch2)
        like_mu2 = self.lambda1 * self.mse_loss(mu1,mu2)
        like_cov2 = self.lambda2 * self.mse_loss(covariance1, covariance2)
        return like_mu2 + like_cov2

    def compute_mean_covariance(self, img):
        batch_size, channel_num, height, width = img.shape
        num_pixels = height * width

        # batch_size * channel_num * 1 * 1
        mu = img.mean(2, keepdim=True).mean(3, keepdim=True)

        # batch_size * channel_num * num_pixels
        img_hat = img - mu.expand_as(img)
        img_hat = img_hat.view(batch_size, channel_num, num_pixels)
        # batch_size * num_pixels * channel_num
        img_hat_transpose = img_hat.transpose(1, 2)
        # batch_size * channel_num * channel_num
        covariance = torch.bmm(img_hat, img_hat_transpose)
        covariance = covariance / num_pixels

        return mu.detach(), covariance.detach()

def id_loss(*alphas):
    loss_id = 0
    for alpha in alphas:
        loss_id += torch.mean(torch.abs(alpha))
    return loss_id