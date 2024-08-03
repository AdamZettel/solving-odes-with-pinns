# A general introduction to solving ODEs with PINNs
I have found a variety of different resources for solving ODEs with PINNs and wanted to collect them here.
## General Note on the PINN approach
- bc stands for boundary condition (this could also be an inital condition) and the physics loss is the difference from the differential equation
$Loss = \lambda_{\text{bc}} \times (L_{\text{boundary}}) + \lambda_{L_{\text{physics}}} \times (\text{physics})$
- in practice, the difficulty of using a PINN is finding the \lambda values as the scale of the boundary loss and the physics loss can vary significantly. In order to get the training to work you may need to vary the lambdas by multiple orders of magnitude $\lambda_{bc} = 100$ and $\lambda_{physics} = 0.01$
