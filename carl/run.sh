#To use this script, first run this to fit your first model:
python kfkd_v1.py fit

#Then train a bunch of specialists that intiliaze their weights from
#your first model:
#modified by carl
#train one 'column' at a time
#we pay additional overhead time of compiling theano in each run
#we gain the benefit of less memory pressure for gpu

python kfkd_v1.py fit_specialists net.pickle 0
python kfkd_v1.py fit_specialists net.pickle 1
python kfkd_v1.py fit_specialists net.pickle 2
python kfkd_v1.py fit_specialists net.pickle 3
python kfkd_v1.py fit_specialists net.pickle 4
python kfkd_v1.py fit_specialists net.pickle 5

#Plot their error curves:
python kfkd_v1.py plot_learning_curves net-specialists_5.pickle
#And finally make predictions to submit to Kaggle:
python kfkd_v1.py predict net-specialists_5.pickle

