1. Loading previously trained models and making them relearn the same dataset will cause them to overtrain.
2. Interestingly enough, the "train" accuracy never really increased with the overtraining, but the "loss" function value did decrease. 
3. This supports the idea that too many iterations on a dataset lead to decreased power for the NN.
4. "Test" has a volatile, rather steep decrease in the accuracy, while the "loss" values are more obvious to an outsider.
5. It appears that more epochs and increased batch size don't really affect the power of the network much, so increasing its accuracy must lie in changing the fundamental way it is constructed. 
6. Furthermore, the number of parameters doesn't change, so the overtraining must be in the parameters focusing too much on a "false" correlation.
