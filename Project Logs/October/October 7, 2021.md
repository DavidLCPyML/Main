# Overview of today: 
## What I did:  
- Started a new project (technically yesterday), applying my current knowledge of AI to generate a program that (theoretically) can break captchas with at least 60% accuracy.
    - The premise: Create a Convolutional Neural Network capable of solving Image captchas of 4 characters or more.
    - Knowledge needed: Deep Neural Networks, Image recognition, Feature extraction, Data generation/processing, Data visualization
## What I now know:
- CAPTCHA, or Completely Automated Public Turing test to tell Computers and Humans Apart, is used to decrease the chances of being overrun by automated scripts or other spam.
- It generally consists of a challenge humans can do, but computers can't, such as the now defunct uses oftext captchas.
    - It is now defunct as a security measure, as the development of **Optical Character Recognition** models (OCR) has been shown since 2014 to be able to crack text CAPTCHAs.
    - It was replaced by reCAPTCHA, but that was also quickly rendered insecure due to the speed of OCR development and Computer Vision(CV).  
- Modules used today:
    - ImageCaptcha, from captcha.image: generates an image captcha given certain text
    - random, a base python library to generate my random captcha strings.
- The text captchas I used employ a series of techniques to prevent computers from viewing the text. 
    - First, a blowing out/in of the individual letters (makes 0, o, O, D, etc. difficult to differentiate)
    - A blurring of the words to make the image harder to scan and interpret by bots (occasionally humans too)
    - A noise curve/line to further obstruct viewing
    - A series of dots to help obfuscate certain features a model might learn.
## What’s next on the list:
- Extract features from the dataset.
    - choices: one-hot encoding (using a matrix of 0s and 1s to determine character type and the position), or segmentation (break captcha into smaller pieces)
- Figure out what type of model I will use. (Probably a CNN)
    - Likely will downsize the dimensions by 2 first, so the input layer should be a 30x80 map of pixels (ImageCaptcha creates 60x160 pixel images).
    - I want my output to be a 62 x n array (n is the number of letters), where each value is the probability that the character the system "sees" is that character.
    - Since it's multiclass one-hot feature scheme, I'll use binary cross-entropy, and probably the adam optimizer
    - activation should be softmax for output, and ReLU for all else.
- Map the history of loss/accuracy, maybe add an activation heatmap if possible so I can see the areas the model is looking at for each letter (not necessary)
- Create the option to save/load model, as well as the ability to upload own captchas to test against model. 
    - Will first check if fixed characters works, then may try my hand at random-length CAPTCHAs.
    - May also benefit from a confusion matrix.
- Modules/Knowledge needed: Segmentation, heatmapping, conversion to one-hot
## Interesting tidbits/thoughts:
- Obviously, the captchas I create will be much harder to recognize for a human than the ones currently used.
    - Some characters I will assume are "hard" are 0 vs o/O/D/ possibly Q.
    - Others include 1, I, and l (lowercase L). The capital "J" may also be confused if the blowing out is strong enough.
    - The number 8 may be mixed up with either 6 or 9, as seen in my MNIST set.
- Accuracy may be depicted as lower, but since it's a product of all the accuracies for each character it's actually more accurate than shown (for example, 90% accuracy on each character means a 65.61% accuracy to guess it all correct, but that is usually not the case.) 
- Color may not play much of a factor, but it helps robustness if the captcha color is not in the dataset (trains it to ignore color in its classification).
## Data/Resources
- https://medium.com/@oneironaut.oml/solving-captchas-with-deeplearning-part-1-multi-label-classification-b9f745c3a599: multilabel classification
- https://medium.com/@oneironaut.oml/solving-captchas-with-deeplearning-part-2-single-character-classification-ac0b2d102c96: single character recognition via segmentation
- https://github.com/danielpontello/cnn-captcha-solving: implementation of captcha solver, "segmentation" style
- https://docs.python.org/3/library/random.html: random() library
- https://www.geeksforgeeks.org/generate-captcha-using-python/: implementing captcha creator
- https://medium.com/analytics-vidhya/how-to-create-your-own-captcha-with-python-3c04e55a5d3f: CAPTCHA tutorial
- http://ceur-ws.org/Vol-1885/93.pdf: CAPTCHA solver with CNN, methods described (no code)
- https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/: one-hot tutorial
- http://themlbook.com/wiki/doku.php: one-hot encoding described here, not sure if it was in "Anatomy of a Learning Algorithm" or "Neural Nets and Deep Learning"
