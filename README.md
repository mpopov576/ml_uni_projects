# ml_uni_projects
Projects from the course “Machine learning and extracting patterns from data” in the Faculty of Mathematics and Informatics, Sofia University.

**Standarts for model reports: **

1. Each row is a hypothesis - a model that was trained and evaluated.
2. The columns are divided into two sets: the first set of columns represent the values of the hyperparameters of the model, the second set: the metrics on the test set. Do not use more than 3 metrics.
3. The first row holds the so-called baseline model. This model can be only one of two things: if currently there is a deployed model on the client's environment, then it is taken to be the baseline model. Otherwise the baseline model is the greediest statistical model. For example, this is the model that predicts the most common class in classification problems.
4. The columns that show the metrics express both the value of the metric as well as the percentage of change compared to the baseline model (we're striving for percentage increase, but should report every case).
5. The rightmost column should be titled Comments and should hold our interpretation of the model (what do we see as metrics, is it good, is it bad, etc). We may include the so-called Error Analysis which details where this model makes mistakes.
6. Above or below the main table there should be a cell that explicitly states which is the best model and why.
7. Below the table or in other sheets there should be the following diagrams: train vs validation metric (the main metric used) and if the model outputs a loss, we should have a train vs validation loss diagram.
8. The table should not be a pandas export - it should be coloured and tell the story of modelling. Bold and/or highlight the entries in which the metric is highest or to which you want to draw attention to.
9. Do not sort the table after completing the experiments - it should be in the order of the created models. This lets you build up on the section Comments and easily track the changes made.
10. Do not create a very wide table - it should be easy to understand which is the best model in one to two seconds of looking at it. Focus on the user experience.
11. Optionally, you could create an additional sheet for the best model in which you put 4-5 examples of correct and incorrect predictions. This will control the client's expectations.
