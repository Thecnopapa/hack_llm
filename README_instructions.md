# Hackathon: Phase 2

We are excited to see your innovative solutions. Please follow the instructions
below to get started with the Phase 2 of the Hackathon!

## Goal

The goal of this phase is to create a model that predicts polution values,
trained from data derived from historic real measurements collected from
automated measurement stations distributed across Barcelona.

Whereas nowadays the word _model_ comes with a strong association to Artificial
Intelligence and Neural Networks, we expect that a number of different
approaches not limited to them can be used or combined to approach the problem.

## Instructions

Follow the steps below to complete this second phase:

1. **Clone the repository**: Clone the repository to your local machine using
`git clone https://github.com/openchip-sw/hack-phase2`.

2. **Implement your solution**: Write your solution within the
`solutionScript.py`. Ensure your code is clean and well-documented. You can use
any extra file, asset or library you consider necessary.

3. **Test your solution**: Use the `testScript.py` (without modifying it) to
evaluate your solution.

4. **Submit your solution**: Submit your solution **before 23/01/2025 at
18:00**. Look at the submission section for more information.

## Data

You have to train your models with the `trainData.csv` file, which includes
information from 5 different measurement stations around Barcelona. The data is
collected every hour and the target variable is the `NO2` value.
The `validationData.csv` file can be used to evaluate your model during
development. The `testData.csv` file, which you don't have acces to, will be
used to evaluate your final solution.

Note that in some cases the input sample may contain a missing measurement
(represented as a `nan` in the callback). This is not an error: it corresponds
to a real-world equipment failure. Decide how you want to handle these cases.
However, make sure you do not include any `nan`s in your predictions: the
evaluation script will skip them if they are present in the gold labels of
`validationData.csv` and `testData.csv`.

Hours go from 1 to 24, in that order. In `trainData.csv`, data are provided
sorted by date, and within each date grouped by station and sorted by hour.
In `validationData.csv` and `testData.csv`, data comes from a single station,
and are provided sorted by date and then hour. For evaluation, each sliding
window of 168h (1 week = 24h * 7 days) will be provided and the prediction for
the next 24h requested.

All data (training, validation, test) comes from the year range 2013 to 2024.

The dataset is derived from the real data collected through the Xarxa de
Vigilància i Previsió de la Contaminació Atmosfèrica and aggregated by the
Departament d'Acció Climàtica, Alimentació i Agenda Rural of the Generalitat de
Catalunya. The data is publicly available under the Generalitat's [Open
Information Use
License](https://administraciodigital.gencat.cat/ca/dades/dades-obertes/informacio-practica/llicencies/).
For more information about the data see the Generalitat's [Open Data
portal](https://analisi.transparenciacatalunya.cat/en/Medi-Ambient/Qualitat-de-l-aire-als-punts-de-mesurament-autom-t/tasf-thgu/about_data).

However, **do not use the data directly from the Generalitat de Catalunya** as
it is not the same as the one provided.

## Implementation

You have to use the `testScript.py` to get the evaluation metric with the
`validationData.csv`. The script will load the `solutionScript.py` and will use
the `predict` function to get the predictions. This predict function will
receive an input sample which consists of 168 (1 week = 24h * 7 days) lists with
the date, hour and NO2 values, for instance:

```
[['2023-01-01', 1, 62.0], ['2023-01-01', 2, nan], ... ['2023-01-07', 24, 60.0]]
```

The output of the function must be a list with 24 values with the NO2
prediction for each hour of the next day (without any date or time
information), for instance:

```
[60.1, 61.4, 62.4, ... 63.0]
```

**IMPORTANT:** Do not modify the `testScript.py` file. This same script will be
used to evaluate your solution with the `testData.csv` and get your final team
score.

The `testData.csv` will have exactly the same structure as the
`validationData.csv`, so make sure that the `testScript.py` works with the
latter to be able to get your evaluation done. It is also important for you to
complete the `requirements.txt` file with all the libraries you have used in
your solution making sure that a fresh environment, just installing the
requirements like `pip install -r requirements.txt`, will be able to run the
`testScript.py` without any problem.

In order to run the scripts, the simplest approach is to set up a _venv_ where
you can install additional Python packages:

```
$ sudo apt install python3-venv
$ python3 -m venv environment
$ source environment/bin/activate
$ pip install -r requirements.txt
$ pip install <package>...
```

## Limitations

Your `predict()` function **cannot make any remote connections**. But it can use
local resources that are part of your submission, so you can be creative with
the assets that you include.

## Evaluation

The evaluation will be done with the typical metric RMSE (root mean squared
error) with the real and predicted values:

$\text{RMSE}(y, \hat{y}) = \sqrt{\frac{\sum_{i=0}^{N - 1} (y_i - \hat{y}_i)^2}{N}}$

You should aim to minimize this metric. The lower the RMSE, the better your
model is. With the default code in the `testScript.py` you should get a RMSE of
around 21.66.

## Submission

Ensure that your solution is well-documented and follows good coding standards.
Once you have completed your implementation, upload it to a GitHub repository
and share the link with [Armando
Rodriguez](mailto:armando.rodriguez@openchip.com)
(armando.rodriguez@openchip.com) and [Edgar
Gonzàlez](mailto:edgar.gonzalez@openchip.com) (edgar.gonzalez@openchip.com) for
evaluation **before 23/01/2025 at 18:00**. If you prefer to keep your repository
private, make sure to give access to it to the
[`arodriguez-oc`](https://github.com/arodriguez-oc) and
[`edgar-oc`](https://github.com/edgar-oc) GitHub users.

Make sure to test your code thoroughly before submission. The repository should
contain:

- Your `solutionScript.py`.
- Any additional code you've developed.
- An updated `requirements.txt` with the libraries you've used.
- Any assets your code needs to load.
- A `SOLUTION.md` file with the following information:
    - Team name
    - Team members
    - Detailed approach followed
    - Main libraries used and why

## Links and Resources

Here are some useful resources and libraries that you may find helpful in
completing this phase of the hackathon. Feel free to use any other resources you
find useful.

- Useful AI and data processing libraries:
    - [Scikit-learn](https://scikit-learn.org/stable/)
    - [Tensorflow](https://www.tensorflow.org/)
    - [Pytorch](https://pytorch.org/)

- Useful time series libraries:
    - [Prophet](https://facebook.github.io/prophet/)
    - [Statsmodels](https://www.statsmodels.org/stable/index.html)
    - [GluonTS](https://ts.gluon.ai/)

- Useful data manipulation libraries:
    - [Pandas](https://pandas.pydata.org/)
    - [Numpy](https://numpy.org/)

- Useful data visualization libraries:
    - [Matplotlib](https://matplotlib.org/)
    - [Seaborn](https://seaborn.pydata.org/)

## Contact

If you have any questions or need further assistance, please reach out to the
project maintainers via email:

* Armando Rodriguez Ramos: <armando.rodriguez@openchip.com>
* Edgar Gonzàlez i Pellicer: <edgar.gonzalez@openchip.com>

Good luck and happy coding!