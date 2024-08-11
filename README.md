# A general introduction to solving ODEs with PINNs
I have found a variety of different resources for solving ODEs with PINNs and wanted to collect them here.
## General Note on the PINN approach
- bc stands for boundary condition (this could also be an inital condition) and the physics loss is the difference from the differential equation, leaving us with: <br>
$Loss = \lambda_{\text{bc}} \times (Loss_{{\text{bc}}}) + \lambda_{{\text{physics}}} \times (Loss_{\text{physics}})$
- In practice, the difficulty of using a PINN is finding the $\lambda$ values as the scale of the boundary loss and the physics loss can vary significantly. In order to get the training to work you may need to vary the $\lambda$ values by multiple orders of magnitude for example: $\lambda_{bc} = 100$ and $\lambda_{physics} = 0.01$. 
