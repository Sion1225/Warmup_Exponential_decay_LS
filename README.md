# Warmup-Exponential decay Learning schedular

This is a learning rate scheduler that was inspired by the Transformer paper, 
_"Attention is All You Need"_ by Ashish Vaswani et al., 2017. 
It uses a warmup step to quickly adapt to the problem through large-scale learning, 
and it can converge to a desired learning rate (target learning rate) through a differentiable function. 
The rate of convergence from the maximum learning rate to the target learning rate can be adjusted by modulating the exponential function (variable a).

### Variables
| Variable | Explain                         |
|----------|---------------------------------|
| max_lr   | maximum learning rate (warm-up) |
| min_lr   | target(minimum) learning rate   |
|num_warmup| number of warm-up steps         |
| a(alpha) | Rate of convergence (curvature of the function): The larger the value, the faster the convergence (0 < a)|

### Graph

![alt text](https://github.com/Sion1225/Warmup_Exponential_decay_LS/blob/main/0.01_50_0.001.png?raw=true)

### Function

$x \leq warmup_{step}$

$$ y=(\max lr/\max warmup_{step})*x $$

$warmup_{step} < x$

$$ y=-e^{\alpha(step-\max warmup_{step})} + \min lr $$