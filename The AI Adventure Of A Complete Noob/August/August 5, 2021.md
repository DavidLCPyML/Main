
# Overview: 
## What I did (over the past month or so):  
- Out of town last week, resumed working on project this week
- Created a model to analyze and predict wine quality of areas near Portugal (https://archive.ics.uci.edu/ml/datasets/wine+quality)
## What I now know:
-  Sometimes data needs to be "fixed"/"realigned" so the interpreter can read the data correctly
  - On the first try, the imported csv squashed all the columns together in one
  - The second try it duplicated the top row, causing the graphs to not plot correctly (no-numeric-data-to-plot error)
  - Fixed by cutting off the top by doing the dataset=dataset[:1] code
  - Further processing was needed, such as changing datatypes to fit the problem requirements
- 
## What’s next on the list:
- 
## Interesting tidbits/my thoughts:
- 
## Data/Resources
- https://archive.ics.uci.edu/ml/datasets/wine+quality - the dataset
- https://machinelearningmastery.com/machine-learning-in-python-step-by-step/ - the site where I found my template
##### Reading List
- https://arxiv.org/abs/2105.04026
- http://cs229.stanford.edu/notes2020spring/cs229-notes1.pdf
