r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


CUSTOM_DATA_URL = "https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip"


def vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['h_dim'] = 256
    hypers['z_dim'] = 128
    hypers['x_sigma2'] = 0.5
    hypers['learn_rate'] = 1e-4
    hypers['betas'] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""

Question 1: Hyperparameter x_sigma2.

Role of the $\sigma^2$ (x_sigma2) hyperparameter:
This parameter represents the variance of the likelihood distribution of the image itself.
In the VAE loss function, it serves as a balancing weight (specifically through the coefficient $\frac{1}{\sigma^2}$) 
between the accuracy of the image reconstruction (reconstruction loss) and maintaining an organized latent space (KL divergence).

Effect of a high value:
The model is penalized less for pixel-level reconstruction errors (since $\frac{1}{\sigma^2}$ becomes small) 
and focuses more on organizing the latent space to match the prior distribution. As a result,
the reconstructed images will be blurrier, but the model will be excellent at sampling, 
making its ability to generate high-quality new images from scratch very good.

Effect of a low value:
The model treats any reconstruction error as critical (since $\frac{1}{\sigma^2}$ becomes very large) 
and focuses solely on copying the original image exactly. As a result, the reconstructed images will be extremely 
sharp and accurate. However, the model will neglect the latent space organization, spreading the representations 
apart and causing overfitting. This severely damages the model's generative capabilities, meaning random sampling 
from the latent space will likely produce incomprehensible noise.

"""

part2_q2 = r"""

Question 2:

2.1. Purpose of the VAE loss terms:

Reconstruction Loss: This term, often represented mathematically as $-\mathbb{E}_{q_\alpha(\mathbf{z}|\mathbf{x})} [\log p_\beta(\mathbf{x}|\mathbf{z})]$,
 ensures that the decoder can accurately rebuild the original input $\mathbf{x}$ from the sampled latent variable $\mathbf{z}$. 
 It forces the network to learn and retain the most meaningful features of the dataset.

KL Divergence Loss: This term, represented as $D_{KL}(q_\alpha(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z}))$, acts as a regularizer. 
It penalizes the model if the posterior distribution produced by the encoder deviates too much from the predefined prior distribution of the latent space, 
which is typically a standard isotropic Gaussian $\mathcal{N}(\mathbf{0}, \mathbf{I})$.


2.2. Effect of the KL loss term on the latent-space distribution:

Without the KL divergence term, the encoder would minimize the reconstruction error by mapping each data point to a tiny, 
isolated, and distant region in the latent space to avoid any overlap (which causes severe overfitting). 
The KL loss prevents this behavior by "pulling" the means ($\mu$) of the encoded distributions toward the origin ($\mathbf{0}$) 
and forcing their variances ($\sigma^2$) close to $\mathbf{1}$. As a result, the latent space becomes a dense, 
continuous, and tightly packed "cloud" where the distributions of similar items smoothly overlap rather than isolating themselves.


2.3. The benefit of this effect:

The primary benefit is that it transforms the autoencoder into a true generative model. 
Because the latent space is densely packed and continuous-meaning there are no "empty holes" where the decoder 
doesn't know what to do - we can sample completely random points $\mathbf{z}$ from the prior distribution $\mathcal{N}(\mathbf{0}, \mathbf{I})$ and feed them to the decoder. 
The decoder will successfully translate these random points into entirely new, 
realistic data instances that never existed in the training set. 
Furthermore, this continuity allows for smooth interpolations between different data points.

"""

part2_q3 = r"""

Question 3:

We start by maximizing the evidence distribution $p(\mathbf{X})$ because it aligns with the fundamental principle of Maximum Likelihood Estimation (MLE). 
The expression $p(\mathbf{X})$ represents the likelihood of observing our true, real-world data under the model we have built. 
Our ultimate goal is for the model to capture the true underlying process that generated the data as accurately as possible.

Why start by maximizing it?

1. Maximum Likelihood Estimation (MLE): 
The logic is straightforward: the best generative model is the one that assigns the highest probability 
to the events that have already occurred in reality. Our training data ($\mathbf{X}$) is an established fact. 
Therefore, we want to optimize the model's parameters so it "believes" the occurrence of this exact training data is highly probable.

2. The Foundation for the ELBO: 
While we theoretically want to maximize $p(\mathbf{X})$ directly, doing so is mathematically and computationally intractable 
because it requires calculating an integral over the entire continuous latent space. 
Because direct calculation is impossible, maximizing $p(\mathbf{X})$ serves as our theoretical "statement of intent". 
From this starting point, we derive a tractable proxy known as the Evidence Lower Bound (ELBO). 

To sum up:
By maximizing this lower bound (which is equivalent to minimizing the VAE loss function), 
we indirectly push up the value of the intractable $p(\mathbf{X})$. 
Therefore, stating the maximization of the evidence distribution is the theoretical bedrock 
from which the entire mathematical derivation of the computable loss function is derived.

"""

part2_q4 = r"""

Question 4:

In a Variational Autoencoder (VAE), we model the log of the latent-space variance,
$\log\sigma^2$, instead of modeling the variance $\sigma^2$ directly due to constraints 
on the neural network outputs and numerical stability during optimization.

Standard output layers of a neural network produce values in the range $(-\infty, \infty)$.
However, a variance parameter must strictly satisfy $\sigma^2 > 0$. 
If the network directly predicted $\sigma^2$, we would have to enforce positivity using activation functions like ReLU or Softplus,
which can cause issues such as dead neurons or exploding gradients. 
By predicting the log-variance, the network can safely output any real number without constraints. 
The true standard deviation needed for the reparameterization trick is recovered using $\sigma = \exp(\frac{1}{2} \log\sigma^2)$, which mathematically guarantees a positive value.

Additionally, the Kullback-Leibler (KL) divergence term in the VAE loss function inherently contains a $\log\sigma^2$ term when computing the distance between the predicted posterior and a standard Gaussian prior. 
Modeling the log-variance simplifies this calculation and prevents numerical instabilities, 
such as taking the logarithm of values close to zero during backpropagation.

"""

# ==============
