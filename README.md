# Exploring TPOT

Let me get to the point directly.

So, I was watching this youtube video in sunday, about how an [AI learns to play Google Chrome Dinosaur Game](https://www.youtube.com/watch?v=sB_IGstiWlc) and
the idea stuck in my head, next day I utilized some of my free time at work to read about it and got to an [article](https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/)</strong> that introduced me to this awesome library [TPOT](https://github.com/EpistasisLab/tpot) (On a funny note, this reminds me to drink Tea, every time I read that)

I started exploring this on Monday and tuesday to see what I could understand and how it can be used in my day to day work, not using right now though, but it does have a lot of potential as mentioned in the article.

At first, it just gave the best fitted pipeline, so I thought what a fancy way to test all the classifier/regressor available in scikit learn(note, TPOT is based on the top on sklearn) using different hyperparameters instead of using grid or random search and choose the best one, reminded of an [H2O demo](https://www.youtube.com/watch?v=42Oo8TOl85I) that I saw earlier, however I haven't explored H2O so this is not a comparison of both of these libraries, it would be interesting to see the results if somebody has. Another library that I came across while going down this road was [auto_sklearn](https://automl.github.io/auto-sklearn/stable/), but we will talk some other day on that and comparison between them.

What I specially like was the feature in TPOT where i can just extract the whole code as a python file, which requires minor changes at my end and get the superb results.
![](https://raw.githubusercontent.com/EpistasisLab/tpot/master/images/tpot-demo.gif)


Next question was if it handles feature selection and feature extraction, how do i find out which one are the feature selected/extracted.
Turns out it does, while there is no direct function to extract, there exist a [workaround](https://github.com/EpistasisLab/tpot/issues/742).

Let's take a sample code to identify the best model (the code is simple and taken directly from repository's [ReadMe file](https://github.com/EpistasisLab/tpot/blob/master/README.md))

```
	from tpot import TPOTRegressor
	from sklearn.datasets import load_boston
	from sklearn.model_selection import train_test_split

	housing = load_boston()
	X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target,
                                                    train_size=0.75, test_size=0.25)

	tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2)
	tpot.fit(X_train, y_train)
	print(tpot.score(X_test, y_test))
	tpot.export('tpot_boston_pipeline.py')
```
The code in `tpot_boston_pipeline.py` looks something like below.

```
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

exported_pipeline = GradientBoostingRegressor(alpha=0.85, learning_rate=0.1,
				loss="ls",  max_features=0.9, min_samples_leaf=5, min_samples_split=6)
exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
```

Don't worry about `'PATH/TO/Data/FILE'`, as long as you are able to split your dataset in train/test like you did earlier for `'TPOT'` you should be good to feed data.

##Links to tutorial
1. [TPOT MNIST pipeline](https://github.com/EpistasisLab/tpot/blob/master/tutorials/MNIST.ipynb)
2. [Portugese Bank Marketing](https://github.com/EpistasisLab/tpot/tree/master/tutorials/Portuguese%20Bank%20Marketing)

There are many others available on the site repository. Do check out the [documentation](http://epistasislab.github.io/tpot/). 
The work is under active developement, look for `help wanted` lables in open issues on git.
