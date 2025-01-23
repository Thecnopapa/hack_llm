# hack_llm


by Iain, Claudia, and Dani

>This is our solution to the problem.
To predict:
- training data should be located in /data/trainData.csv
- Run testScript.py from the ../scripts/ directory (or won't find the data and other scripts)
> Optional

  - in testScript.py: 
    - replace **predict(windowToPredict)**
    - to **predict(windowToPredict, simple=False)** to see our model (not) work


Apologies if not all the code is not well commented or you see weird stuff.



The scripts containing our implemented code are:
- 
- scripts/process.py (data processing)
- dataload (everything to put the training data into the model)
- scripts/model.py (contains our model)
- scripts/train (stuff to train the model)

Everything else is mostly debris