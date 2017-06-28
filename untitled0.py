from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import time


####N IS WHAT IS BEING CHANGED HERE, VARYING VALUES IN THE N REPITITION
####
def graphGen(n_folds):
    start_time = time.time()
    dataaa = datasets.load_boston() #BOSTON HOUSE PRICES FROM THE INSTITUTE OF ()
    #OR USE THIS FOR DIABETES DATA
    #dataaa = datasets.load_diabetes()
    
    X = dataaa.data[:150]
    y = dataaa.target[:150]
    
    lasso = Lasso(random_state=0)
    allAlphas = np.logspace(-4, -0.5, 30)
    
    scores = list()
    stdrdScores = list()

    for alpha in allAlphas:
        lasso.alpha = alpha
        this_scores = cross_val_score(lasso, X, y, cv=n_folds, n_jobs=1)
        scores.append(np.mean(this_scores))
        stdrdScores.append(np.std(this_scores))
    
    scores, stdrdScores = np.array(scores), np.array(stdrdScores)
    
    plt.figure().set_size_inches(8, 6)
    plt.semilogx(allAlphas, scores)
    
    standarderror = stdrdScores / np.sqrt(n_folds)
    
    plt.semilogx(allAlphas, scores + standarderror, 'b--')
    plt.semilogx(allAlphas, scores - standarderror, 'b--')
    
    plt.fill_between(allAlphas, scores + standarderror, scores - standarderror, alpha=0.2, color='red')
    
    plt.ylabel('Score +- standard error, which ideally approaches 0 and doesnt deviate away from 0 as alpha does.')
    plt.xlabel('alpha')
    plt.axhline(np.max(scores), linestyle='--', color='.8')
    plt.xlim([allAlphas[0], allAlphas[-1]])
    plt.savefig(str(n_folds) + ".png")
    
    print(str(n_folds) + " takes " + (time.time() - start_time))

for i in range(2, 100):
    graphGen(i)
    #The graphGen function takes in the 'k' in k-folds and prints out AND puts out a image file of the graph.