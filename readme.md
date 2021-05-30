# LinFit

Linfit is a program which trains trains ReLU NNs with linear regression. It isn't super practical since it doesn't outperform gradient based methods ands still is a serial process. Since this was a weekend project, there could be a lot more formalization in the mathematics behind this idea.

## How
LinFit exploits that the ReLU activation is approximately linear and the solution to the weight matrix can be approximated with linear regression. The inconsistencies create non linearities which will help the next layers learn.
The algorithm to update a layer’s ($L_a$) weights: 

1. Forward the network up until $L_{a-1}$
   1. This allows us to feed in non-linear features into the layer which will make the layers job ‘easier’
2. Do a reverse approximation back through the network. First multiply all the weight matrices after the network ($L_{a+1} \cdot L_{a+2} \cdot L_{a+3} \ldots L_{N}$ where $N$ is the number of layers) and then do a Penrose inverse.
   1. Since the weight matrices are usually non-square (non invertible too) I used the Penrose inverse which approximates the solutions if the matrix is over constrained, and one possible solution (the probability of it being correct is zero since that is a point, however since it’s linear 

