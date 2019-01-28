# predictorv3

A logistic regression machine learning element to predict result of a match. 
Followed  by taking the average xG scored and conceded by both the home and away team to produce an average value for the home team to score and away team to score 
which then uses a Poisson distribution with these numbers as mean to find the most likely score line with matching result to the first element.
