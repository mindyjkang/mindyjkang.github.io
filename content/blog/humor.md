+++
title = '[Review] HuMoR: 3D Human Motion Model for Robust Pose Estimation'
date = 2024-04-21T21:27:42+09:00
draft = false
+++

## Background

### Bayesian Statistics

given data samples $\{ x^{(1)}, \dots , x^{(m)} \}$ & prior $p(\theta )$ we can recover posterior distribution $p(\theta | x^{(1)}, \dots , x^{(m)})$

$$
p(\theta | x^{(1)}, \dots , x^{(m)}) = {p(x^{(1)}, \dots , x^{(m)} | \theta) \cdot p(\theta) \over p((x^{(1)}, \dots , x^{(m)})}
$$

### ELBO (Evidence Lower Bound)

Given probabilistic model $p()$ , latent variables $\boldsymbol{h}$ and observed data $\boldsymbol{v}$, to model the data distribution $p(\boldsymbol{v})$ 

$$
p(\boldsymbol{v}) = {p(\boldsymbol{v},\boldsymbol{h})\over p(\boldsymbol{h}|\boldsymbol{v})}
$$

 we are interested in computing  $p(\boldsymbol{h}|\boldsymbol{v})$ (probability distribution over the latent space given the data), and consequently $log ~p(v;\theta)$ in an optimization problem. But it can have high computational cost. Instead we try to approximate it by fitting a approximate posterior $q$ with a lower bound which is called *ELBO* ($L$).

$$
p_{\theta}(\boldsymbol{h}|\boldsymbol{v}) \approx q_{\varphi}(\boldsymbol{h}|\boldsymbol{v}) 
$$
$$
L(v, \theta, q) = log ~p_{\theta}(v) - D_{KL}(q(h|v)||p_{\theta}(h|v))
$$

The goal is to minimize KL divergence score between $q(h|v)$ and $p_\theta(h|v)$

If we rearrange ELBO equation

$$
L(v, \theta, q) = log ~p_\theta(v) - D_{KL}(q(h|v)||p_\theta(h|v)) \\\ 
= - E_{h\sim q}[log~q(h|v) - log~p_\theta(h, v)] \\\ 
= E_{h\sim q}[log~p_\theta(h,v)] + H(q)
$$

if we find $q(h|v)$  that approximates $p(h,v)$  well, the lower bound will be narrow and $L(v, \theta, q)$ will be close to $log~p(v)$

### VAE (Variational Autoencoders)

VAE is a type of autoencoder that trains encoder $q(z|x)$ to output set of parameters that shape the prior distribution $p_{model}(z)$. The latent $z \sim p_{model}(z)$ is sampled from the prior distribution and generates the output through decoder $p_{model}(x|z)$. 

Like other autoencoders we train by maximizing the variational (evidence) lower bound (ELBO) $L(q)$ associated with data point $x$

![Untitled](/humorimages/Untitled.png)

In order to maximize *ELBO*, reconstruction term $E_{z\sim q(z|x)}[log p_{model}(x|z)]$ needs to be maximized and KL divergence $D_{KL}$ term should be minimized.

In practice, multivariate Gaussian distribution is usually chosen for $p_{model}(x; g(z)) = N(x|\mu, \sigma)$. Prior distribution in VAEs captures the probabilistic distribution of data in predefined shape of latent space. 

## HuMoR

HuMoR (3D Human Motion Model for Robust Estimation of temporal pose and shape) is a VAE based generative model that models temporal difference between 3d human pose state representations. It can be used to refine ambiguous pose estimates into more plausible human motion sequence with learned motion distribution. 

### State Representation

Training dataset (AMASS) is processed into state of moving person matrix $x$, which is consisted of root translation ( $r$ ), root orientation ( $\Phi$ ), body pose joint angles $\Theta \in \R^{3 \times 21}$ and joint positions $J \in \R^{3 \times 22}$, $\dot{r}, \dot{\Phi}, \dot{J}$  denotes velocities of each parameters.

$$
x = [ ~r~ \dot{r}~\Phi~\dot{\Phi}~\Theta~J~\dot{J} ~]
$$

### Latent Variable Dynamics Model

![Untitled](/humorimages/Untitled%201.png)

HuMoR incorporates **CVAE (Conditional Variable Autoencoder)** to model the latent distribution of state transition. CVAE is a VAE with conditional property added to the prior distribution. HuMoR uses previous motion state $x_{t-1}$ as conditional variable, and conditional prior is defined as below.

$$
p_\theta (z_t|x_{t-1}) = N(z_t; ~\mu_\theta(x_{t-1}), \sigma_\theta(x_{t-1}))
$$

Also decoder $p_\theta(x_t|z_t, x_{t-1})$ is also conditioned on previous state ($x_{t-1}$) to output $\Delta_\theta$ and $\hat{c}_t$. $\Delta_\theta$ is change in state that forms current state estimate ($x_t$) that is predicted by the model. 

$$
x_{t} = x_{t-1} + \Delta_\theta(z_t, x_{t-1}) + \eta , ~~~ \eta \sim N(0, I)
$$

and $\hat{c}_t$ is person-ground contact vector which is the probability that each of 8 body joints is in contact with the ground at time $t$.

CVAE architecture is consisted of 3 components; encoder (posterior) , conditional prior and decoder. All 3 networks are 4 or 5 MLP layers with ReLU activations.

CVAE is trained with maximizing ELBO like other VAEs.

$$
log~p_\theta(x_t|x_{t-1}) \ge E_{q_\phi}[log~p_\theta(x_t|z_t,x_{t-1})] - D_{KL}(q_\phi(z_t|x_t, x_{t-1})||p_\theta(z_t|x_{t-1}))
$$

consequently the loss term is 

$$
L_{rec}+w_{KL}L_{KL}+L_{reg}
$$

Reconstruction loss $L_{rec}$ is L2 loss between input data and decoder output, $L_{KL}$ is KL divergence term and $L_{reg}$ is additional regularization term. 

### Test-time Motion Optimization

![Untitled](/humorimages/Untitled%202.png)

Trained HuMoR model can be used to refine the state trajectories of human pose predictions based on various modalities. Gaussian Mixture Model is used to model the initial state representation separately. Performing inference test time, conditional prior takes in input state $x_{t-1}$ and samples latent code $z_{t}$ from the latent distribution. Decoded output $\hat{x}_{t}$ is used as input again to perform autoregressive rollout.

Refinement process is a optimization with an objective to incorporate HuMoR components to match predicted human pose estimates. We optimize initial state $x_0$, latent transitions $z_{1:T}$, ground plane $g$, and SMPL shape parameter $\beta$ using below objective term based on MAP (Maximum a-posteriori) estimate.

$$
\underset{x_0,z_{1:T},g,\beta}{max} p(x_0,z_{1:T},g,\beta|y_{0:T})\\\\ 
= \underset{x_0,z_{1:T},g,\beta}{max} p(y_{0:T}|x_0,z_{1:T},g,\beta)p(x_0,z_{1:T},g,\beta) \\\\
=\underset{x_0,z_{1:T},g,\beta}{max}~logp(y_{0:T}|x_{0}, z_{1:T}, g,\beta) + logp(x_0, z_{1:T}, g, \beta) \\\\
= \underset{x_0,z_{1:T},g,\beta}{min}~-\sum^T_{t=0}logp(y_t|x_t, \beta) - \sum^T_{t=1}logp(z_t|x_{t-1}) \\\\ - logp(x_0|g) - logp(g) - logp(\beta) \\\\
= \underset{x_0,z_{1:T},g,\beta}{min}~E_{data}+E_{CVAE}+E_{init}+E_{ground}+E_{shape}
$$

- Assume $y_t$ is only dependent on $x_0$ and $z_{\le t}$, and following CVAE rollout $x_t = f(x_0, z_{1:t})$

$$
\because p(y_{0:T}|x_{0}, z_{1:T}, g,\beta) = p(y_0|x_0, \beta)\prod^T_{t=1}p(y_t|z_{\le t},x_0, g, \beta) = \prod^T_{t=1}p(y_t|x_{t}, \beta) \\\\
 p(x_0, z_{1:T}, g, \beta) = p(x_0, g, \beta)\prod^T_{t=1}p(z_t|z_{<t}, x_0, g, \beta) = p(x_0|g)p(g)p(\beta)\prod^T_{t=1}p(z_t|x_{t-1}) 
$$

- replacing $E_{mot} = E_{CVAE} + E_{init}$ and $E_{reg} = E_{ground} + E_{shape} + ...$ 

final optimization objective is

$$
\underset{x_0,z_{1:T},g,\beta}{min}~E_{mot}+E_{data}+E_{reg}
$$