# Overview of today: 
## What I did:  
- Completed the project:
    - implemented one-hot encoding of the labels as well as processing the images.
    - Implemented graphs showing accuracy and loss for validation and train sets
    - Implemented user uploaded checking
    - Compiled and saved completed model. 
- Read a paper: Batch normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
## What I now know:
- How to implement one-hot encoding, as well as how to handle input/output of these label types without using Sequential() in keras (wasn't able to find a way to get it to work).
##### Primarily Batch Normalization
- **ICS**, also called Internal Covariate Shift, is caused when small changes in hidden layers snowball into a vanishing/exploding gradient problem the deeper the network is.
    - It occurs when parameter changes lead to changes in input distributions of successive layers, causing and amplifying effect as the network proceeds to go deeper.
    - This also means sigmoid and tanh activation are difficult to use, as the saturation of the network due to changing params causes vanishing gradient.
    - The paper proposes that if we are able to ensure the distribution of nonlinearities remains stable, then we would be able to use faster learning rates, allowing us to reach convergence quicker. 
- The authors claim that while whitening layers (i.e. setting weights to 0 each layer) would work, ultimately it opens the can of worms regarding the exploding gradient as that ignores the dependence of the weighted sum on the bias, causing no change in loss but ever increasing bias (meaning exploding gradient).
    - Instead, they propose that we normalize activations to a normal distribution, allowing us to parameterize the mean and variance.
    - This works since the mini-batches produce estimates of the distribution of each activation, of which the mean and stdev can be optimized to best fit ideal activations.
- \hat{x}_{i} = \frac{x_{i} - \mu_{\mathcal{B}}}{\sqrt{\sigma^{2}_{\mathcal{B}}+\epsilon}}
    - Therefore, we only need to use GD to optimize gamma and beta, allowing for reduced ICS and faster training.
- Findings included:
    - BN makes the use of sigmoid or tanh activation now viable
    - Dropout is no longer needed to prevent ICS
    - BN allows for massively boosted learning speed as well as increased accuracy as better extrema are achieved with the normalization.
    - BN works especially well on CNNs and image recognition.
## What’s next on the list:
- Continue reading literature
    - Try going deeper, see if techniques like RNNs are still used
## Interesting tidbits/thoughts:
##### CAPTCHA Breaker findings:
- This worked extremely well, it managed to differentiate between many characters, only occassionally getting some letters mixed up.
    - As expected, commonly mistaken characters were 0/o/O/D and 1/l/I, something I think I would be hard pressed to differentiate if I were given this CAPTCHA.
    - Otherwise, it was surprisingly accurate at predicting, achieving an approximate accuracy of 70%.
- Interestingly, the first and last digit were predicted more accurately than the middle digits.
    - On the model I tested, it was the second and third characters that had a lower accuracy.
    - However, I would conclude that it is true for any length captcha, as the model needs to "count" the position it is at, and the randomness of the characters on the captcha means that accuracy will be slightly lower than that of the edge characters.
    - 6 and 9 were also easily differentiated, something surprising to me.
##### Batch Normalization thoughts/findings:
- Doing some further research on ICS and BNorm, I was surprised to find that there isn't a consensus on why BN works so well. Theories include:
    - It smoothes the optimization curve, allowing for the parameters to be less volatile and therefore quicked GD.
    - The normalizing of the hidden input layers that BN does helps reduce interdependencies, allowing for quicker tweaking of parameters, leading to faster training/convergence.
    - Supposedly, it does indeed reduce ICS, but apparently ICS was not the primary reason to BN's affect on training.
- This does lead to some questions:
    - How does the beta and gamma params affect smoothness of the curve?
    - Does BN affect generalizability?
- One thing I found interesting here is that the consensus of **when** to use BN is also not concrete either.
    - Some say you apply it before the nonlinearity, others claim it works better after the nonlinearity.
## Data/Resources
- https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/: one-hot encoding walkthrough
- https://arxiv.org/abs/1502.03167: Batch normalization paper
- https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338: Article on Batch Normalization
- https://towardsdatascience.com/batch-normalization-explained-algorithm-breakdown-23d2794511c: another article on BN
