
# Overview: 
## What I did (over the past month or so):  
- Out of town last week, resumed working on project this week
- Created a model to analyze and predict wine quality of areas near Portugal (https://archive.ics.uci.edu/ml/datasets/wine+quality)
  - out of a set of 6 algorithms, Decision trees were picked due to performing the highest on the pretests.
## What I now know:
-  Sometimes **data needs to be "fixed"/"realigned" before processing** so the interpreter can read the data correctly
  - On the first try, the imported csv squashed all the columns together in one
  - The second try it duplicated the top row, causing the graphs to not plot correctly (no-numeric-data-to-plot error)
  - Fixed by cutting off the top by adding the "dataset=dataset[:1]" line of code
  - Sometimes **datatype modification is also necessary**, such as changing datatypes to fit the problem requirements
- Box-and-whisker plots will usually show clumps of "outliers" depending on the type of distribution
  - Some plots were relatively **uniformly normally distributed and thus had less outlier clumps**, i.e. citric acid, quality, and alcohol
  - One plot, density, followed a **normal distribution (clumps on both sides)**
  - Other plots, such as Residual sugar, chlorides, sulphates, and the rest to some degree had pretty **heavy clumps on one side, indicating a moderately/heavily skewed distribution.**
  - **Histograms also functioned similarly**, depicting how skewed each feature was distributed.
- The plots with fewest clumps indicate that the data is as close as possible to the perfect uniform normal distribution.
  - in essence, boxplots don't indicate outliers so much as they indicate the shape of the distribution.
- Sometimes you need to perform multiple analyses to figure out the best algorithm to use for a certain problem
  - ex: Using the 6-ish most simple classifiers (logreg, dtree, KNN, SVM, LDA/linreg, and Naive Bayes/GaussianNB) to generate grouped performance and pick the best
        - Sometimes, more complex models can/will be necessary such as NNs (of course assuming it's SL and not USL/SSL)
## What’s next on the list:
- Attempt a non-supervised or Semi-supervised learning project (so far SL and NNs/DNNs have been explored, SSL/USL/RL has not yet)
## Interesting tidbits/my thoughts:
- Some columns were indeed related as expected, like fixed acidity/citric acid and total SO2/free SO2 
  - They made sense intuitively i.e. more citric acid = higher acidity, more free SO2 = more total SO2
- Some columns were less intuitive, like density/fixed acidity or density/alcohol
- The number of samples with value quality '3' was almost nonexistent, so values in the precision-recall table were 0 (got the prediction wrong, 0/1 = 0)
  - with more samples, a nonzero real precision/recall value would've been possible, highlighting the need for large datasets when performing such analyses.
  - accuracy (bounds) tended to lie in intervals [0.63, 0.66], which isn't terrible but also isn't very good, could be because of some variables being correlated with each other
- This project was also the first time I had to consider hyperparameters (like how many splits were needed to reduce the entropy)
## Data/Resources
- https://archive.ics.uci.edu/ml/datasets/wine+quality - the dataset
- https://machinelearningmastery.com/machine-learning-in-python-step-by-step/ - the site where I found my template
- https://swampthingecology.org/blog/too-much-outside-the-box-outliers-and-boxplots/ - source about boxplots
##### Reading List
- https://arxiv.org/abs/2105.04026
- http://cs229.stanford.edu/notes2020spring/cs229-notes1.pdf
