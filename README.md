# Implementation of Non-Linear Independent Component Estimation (NICE & RealNVP) in TF-2 
This repository presents an implementation of the NICE model, as described in the 2014 paper authored by Laurent Dinh,
David Krueger, and Yoshua Bengio. The NICE model serves as the foundational layer for subsequent normalizing flow models.
### Model Explenantion
#### Main Components
The main idea behind the model is to:
1. Transform the inital unknown data distribution into a latent space with a known density via an invertible function.
2. Train the model by maximizing the known likelihood of the mapped data distribution by the change-of-variable rule.
3. Sample from the known density and invert the sampled points to reconstruct the original data space. 

The following is the mathematical implementation of the previously discussed process:
1. Define the "latent" hidden space known distribution as the product of independent Logistic or Gaussian univariate densities: </br>
```math
$$ \mathbf{h} \sim  p_{H}(\mathbf{h}) = \prod_{i}{p_{H_{i}}(h_{i})}  $$
```
2. Map the initial data distribution to the hidden space distribution via $f$, parametrized by the parameters $\theta$: </br>
  + Compute the latent representation and density
```math
$$ f: \mathbf{X} \rightarrow \mathbf{H} \Rightarrow f_{\theta}(x) = h $$
```

```math
f^{-1}: \mathbf{H} \rightarrow \mathbf{X} \Rightarrow f^{-1}_{\theta}(h) = x $$
```

```math
 p_{\mathbf{X}}(\mathbf{x}) =   p_{\mathbf{H}}(f_{\theta}(\mathbf{x})) | det(\frac{\partial f_{\theta}(\mathbf{x}) }{\partial x}) | $$
```
  + Compute the likelihood of the latent space variables $h$ via the change of
   variable formula: </br>
```math
 \mathcal{L} ( p_{\mathbf{X}}(\mathbf{x}))  =  \sum_{i} log (p_{\mathbf{H_{i}}}(f_{\theta}(\mathbf{x}_{i}))) +  log (| det(\frac{\partial f_{\theta}(\mathbf{x}) }{\partial x}) |) $$
```

3. Samples of the initial data distribution are computed by inverting the samples from the hidden space distribution:
```math
 \mathbf{h} \sim  p_{H}(\mathbf{h}) 
```
```math
 \mathbf{x} \sim  f^{-1}(\mathbf{h})
```
#### Coupling function (Additive Coupling)
Since $f$ must be invertible in order to evaluate the likelihood, update the parameters, and invert the samples from the base prior distribution, the authors choose to implement 
and additive coupling rule which takes the following form: 
1. Partition the initial data space into two partitions $x_{a}\in\mathbb{R}^{D-b}$ and $x_{b}\in\mathbb{R}^{D-a}$
2. Apply a transformation $g$ only on one partition:
```math
 h_{a} = x_{a} 
```
```math
 h_{b} = x_{a} + g_{\theta}(x_{a}) 
```
The inverse of this coupling function will be:
```math
 x_{a}  = h_{a}
```
```math
 x_{a} = h_{b} - g(x_{a}) 
```
The jacobian of this function is lower triangular and has unit determinant since:
```math
\mathbb{J} =
\begin{bmatrix}
 \frac{\partial{h_{a}}}{\partial{x_{a}}} & \frac{\partial{h_{a}}}{\partial{x_{b}}} \\ 
 \frac{\partial{h_{b}}}{\partial{x_{a}}} & \frac{\partial{h_{b}}}{\partial{x_{b}}} \\ 
\end{bmatrix} =

\begin{bmatrix}
 \mathbf{I} & \mathbf{0}\\ 
 \frac{\partial{h_{b}}}{\partial{x_{a}}} & \mathbf{I} \\ 
\end{bmatrix} 
```
and the resulting determinant is:
```math
det(\mathbb{J}) =  \mathbf{I} \cdot \mathbf{I}  + \mathbf{0} \cdot \frac{\partial{h_{b}}}{\partial{x_{a}}} = \mathbf{I}
```
```math
log(det(\mathbb{J})) =  log(\mathbf{I}) = 0
```

#### Scaling function
To make the function more flexible the authors propose to multiply the output of the final coupling transformation with an invertible function which is applied element wise:
```math
y_{i} = g_{\theta_{i}}(x_{i}) = x_{i} \cdot e^{\theta_{i}}
```
```math
x_{i} = g^{-1}_{\theta_{i}}(y_{i}) = y_{i} \cdot e^{-\theta_{i}}
```
The jacobian of this function is diagonal and the resulting determinant is the product of the diagonal components:
```math
\mathbb{J} =
\begin{bmatrix}
 \frac{\partial{y_{a}}}{\partial{x_{a}}} & \frac{\partial{y_{a}}}{\partial{x_{b}}} \\ 
 \frac{\partial{y_{b}}}{\partial{x_{a}}} & \frac{\partial{y_{b}}}{\partial{x_{b}}} \\ 
\end{bmatrix} =

\begin{bmatrix}
 e^{\theta_{11}} & \mathbf{0}\\ 
 \mathbf{0} & e^{\theta_{ii}} \\ 
\end{bmatrix} 
```
```math
det(\mathbb{J}) =  \prod_{i} e^{\theta_{ii}}
```
```math
log(det(\mathbb{J})) =  \sum_{i}\theta_{ii}
```

##### Coupling Function (Affine Coupling)
In the paper [RealNVP](https://arxiv.org/abs/1605.08803) the authors combined the addition and scaling couplings to jointly learn to translate and scale the base density space with input dependent translation and scaling parameters. 
The coupling takes the following form: 
```math
h_{a} = x_{a} 
```
```math
h_{b} = x_{b}  \cdot exp(s_{\theta}(x_{a}))  + g_{\theta}(x_{a}) 
```
The inverse coupling function will be:

```math
x_{a} = h_{a}
```
```math
x_{b} = (h_{b} - g_{\theta}(x_{a})) \cdot exp(-s_{\theta}(x_{a}))
```
where $g_{\theta}$ and $s_{\theta}$ are neural networks. <br/>
The jacobian of this function is lower triangular and has unit determinant since:
```math
\mathbb{J} =
\begin{bmatrix}
 \frac{\partial{h_{a}}}{\partial{x_{a}}} & \frac{\partial{h_{a}}}{\partial{x_{b}}} \\ 
 \frac{\partial{h_{b}}}{\partial{x_{a}}} & \frac{\partial{h_{b}}}{\partial{x_{b}}} \\ 
\end{bmatrix} =

\begin{bmatrix}
 \mathbf{I} & \mathbf{0}\\ 
 \frac{\partial{h_{b}}}{\partial{x_{a}}} & diag(exp(s_{\theta}(x_{a})) \\ 
\end{bmatrix} 
```
```math
det(\mathbb{J}) =  \mathbf{I} \cdot diag(exp(s_{\theta}(x_{a})) =  diag(exp(s_{\theta}(x_{a}))
```
```math
log(det(\mathbb{J})) =  \sum_{i} s_{\theta}(x_{a})_{i}
```
### References

For further details, please refer to the [NICE paper](https://arxiv.org/abs/1410.8516).

## Results 
### Circle Dataset 
#### NICE Results
![NICE Model Samples Circle](https://github.com/claCase/NormalizingFlow/blob/master/figures/circles/NICE%20(Trained)%20-%20True%20vs%20Model%20Samples.png)
![NICE Model Density Circle](https://github.com/claCase/NormalizingFlow/blob/master/figures/circles/NICE%20-%20Samples%20from%20Trained%20Model.png)
#### RealNVP Results
![RealNVP Model Samples Circle](https://github.com/claCase/NormalizingFlow/blob/master/figures/circles/RealNVP%20(Trained)%20-%20True%20vs%20Model%20Samples.png)
![RealNVP Model Density Circle](https://github.com/claCase/NormalizingFlow/blob/master/figures/circles/RealNVP%20-%20Samples%20from%20Trained%20Model.png)

### Half-Moons Dataset 
#### NICE Results
![NICE Model Samples Half Moons](https://github.com/claCase/NormalizingFlow/blob/master/figures/moons/NICE%20(Trained)%20-%20True%20vs%20Model%20Samples.png)
![NICE Model Density Half Moons](https://github.com/claCase/NormalizingFlow/blob/master/figures/moons/NICE%20-%20Samples%20from%20Trained%20Model.png)
#### RealNVP Results 
![RealNVP Model Samples Half Moons](https://github.com/claCase/NormalizingFlow/blob/master/figures/moons/RealNVP%20(Trained)%20-%20True%20vs%20Model%20Samples.png)
![RealNVP Model Density Half Moons](https://github.com/claCase/NormalizingFlow/blob/master/figures/moons/RealNVP%20-%20Samples%20from%20Trained%20Model.png)

### Spirals Dataset 
#### NICE Results 
![NICE Model Samples Sprials](https://github.com/claCase/NormalizingFlow/blob/master/figures/spirals/NICE%20(Trained)%20-%20True%20vs%20Model%20Samples.png)
![NICE Model Density Spirals](https://github.com/claCase/NormalizingFlow/blob/master/figures/spirals/NICE%20-%20Samples%20from%20Trained%20Model.png)
#### RealNVP Results 
![RealNVP Model Samples Spirals](https://github.com/claCase/NormalizingFlow/blob/master/figures/spirals/RealNVP%20(Trained)%20-%20True%20vs%20Model%20Samples.png)
![RealNVP Model Density Spirals](https://github.com/claCase/NormalizingFlow/blob/master/figures/spirals/RealNVP%20-%20Samples%20from%20Trained%20Model.png)
