# LinFit

Linfit is a program which trains trains ReLU NNs with linear regression. It does outperform single step gradient descent a majority of the time with popular algorithms (SGD, Adagrad, Adam) with default settings. It seems to be significantly faster than any other popular algorithm tested. However, this algorithm only works with pure ReLU NNs.

## How
LinFit exploits that the ReLU activation is approximately linear and the solution to the weight matrix can be approximated with linear regression.
The algorithm to update a layer’s ($L_a$​) weights: 

1. Forward the network up until $L_{a-1}$​​ the result will be called $x$​.
   1. This allows us to feed in non-linear features into the layer which will make the layers job ‘easier’
2. Do a reverse approximation back through the network. This step assumes that layers $L_{a+1}...L_N$​ have no ReLU activation and are purely linear. First all the weight matrices after the layer ($L_{a+1} \cdot L_{a+2} \cdot L_{a+3} \ldots L_{N}$​ where $N$​ is the number of layers) are multiplied and then a Penrose inverse is used. This result will be called $y$​.
   1. Since the weight matrices are usually non-square (non invertible too) the Penrose inverse is used. The Penrose inverse approximates the solutions if the matrix is over or under constrained. In the case where the matrix is under constrained the Penrose inverse will select a possible solution.
3. Now we have essentially propagated the network’s input data up to the layer right before ($x$) and estimated the expected input for the layer right after ($y$). If we assume the layer has no activation function and is linear, then we have the output of the previous layer and the expected input for the next layer so all we have to do is solve the matrix equation $Ax=y$​. This is a classic case of linear regression and we can use the .

