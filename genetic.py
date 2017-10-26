import pandas as pd
import numpy as np
import math
import random

def input_split(X,foldno):
    traintuples = []
    testtuples = []
    for i in range(X.shape[0]):
        if i%10 == foldno:
            testtuples.append(X[i])
        else:
            traintuples.append(X[i])

    return np.array(traintuples),np.array(testtuples)


def std_calc(lists,mean):
    diffsqsum = 0
    for i in range(len(lists)):
        diffsqsum += (lists[i] - mean)**2
    return round(math.sqrt(diffsqsum/float(len(lists))),3)

def gaussian_func(val,mean,std):
    return ((1/math.sqrt(2*math.pi*(std**2)))) * math.exp(-((val-mean)**2.0)/(2*(std**2)))

def naive_bayes(X,Y,YT,Z):

    tp=0
    tn=0
    fn=0
    fp=0

    #print X.shape,Y.shape,Z.shape,YT.shape
    #print Y
    meanyes = []
    meanno = []

    stdyes = []
    stdno = []

    #print gaussian_func(3,4,5)
    #print "---------------"
    #print Y
    for j in range(X.shape[1]):
        yeslist = []
        nolist = []
        for i in range(0,X.shape[0]):
            if Y[i]=="Yes":
                #print "yes is there"
                yeslist.append(X[i,j])
            else:
                #print i
                nolist.append(X[i,j])
        #print nolist
        meanyes.append(round(sum(yeslist)/float(len(yeslist)),2))
        meanno.append(round(sum(nolist)/float(len(nolist)),2))

        stdyes.append(std_calc(yeslist,meanyes[-1]))
        stdno.append(std_calc(nolist,meanno[-1]))

    #print Z.shape
        YL = list(Y)
        a = YL.count("Yes")
        b = YL.count("No")
        pyes = float(a)/a+b
        pno = float(b)/a+b

    for j in range(0,Z.shape[0]):
        #change
        probyes = pyes
        probno = pno
        for i in range(X.shape[1]):
            probyes *= gaussian_func(Z[j,i],meanyes[i],stdyes[i])
        for i in range(X.shape[1]):
            probno *= gaussian_func(Z[j,i],meanno[i],stdno[i])
        classlabel = "Yes" if probyes>probno else "No"

        if YT[j]=="Yes":
            if classlabel == "Yes":
                tp+=1
            else:
                fp+=1
        else:
            if classlabel == "No":
                tn+=1
            else:
                fn+=1

        #print YT[j]+"\t"+classlabel

    acc = (tp+tn)/float(tp+tn+fp+fn)
    precision=0
    recall=0

    try:
        precision = float(tp)/(tp+fp)
    except Exception as e:
        a=0

    try:
        recall = float(tp)/(tp+fn)
    except Exception as e:
        a=0

    return acc,precision,recall
    #print probyes,probno

def init_population():
    population = np.random.randint(2,size=(30,44))
    return population

def objective_func(A,test_chromosome):
    X=[]
    #print test_chromosome[0]
    for i in range(44):
        if test_chromosome[i]==1:
            X.append(A[:,i])
    X.append(A[:,44])
    X = np.array(X).transpose()
    VADE = X
    #print VADE.shape
    sumacc=0
    sumprec=0
    sumrec=0
    for i in range(10):

        train_data,test_data = input_split(VADE,i)

        Y = train_data[:,VADE.shape[1]-1]
        YT = test_data[:,VADE.shape[1]-1]
        XT = test_data[:,0:VADE.shape[1]-1]
        X = train_data[:,0:VADE.shape[1]-1]
        #print Y
        acc,precision,recall = naive_bayes(X,Y,YT,XT)
        sumacc+=acc
        sumprec+=precision
        sumrec+=recall
    #print str(acc)+"\tFold no: {}".format(i+1)
    return sumacc/10.0,sumprec/10.0,sumrec/10.0
        #print str(acc)+"\tFold no: {}".format(i+1)

    #return np.array(X).transpose()

def fitness_func(A,population):
    global max_acc
    global maxprecision
    global maxrecall
    global chromosome
    fitness_list = []
    maxacc=0
    for i in range(30):
        acc,precision,recall = objective_func(A,population[i])

        if acc > maxacc:
            maxacc = acc
            maxprecision = precision
            maxrecall = recall
            chromosome = population[i]

        if acc > max_acc:
            max_acc = acc
        #print "Accuracy = "+str(acc)+"\t\tPrecision = "+str(precision)+"\t\tRecall = "+str(recall)+"\t\tfor chromosome {}".format(i+1)
        fitness_list.append(acc)
    #print fitness_list
    #print "Maximum accuracy = {}".format(maxacc) + " Precision = {}".format(maxprecision) + " Recall = {}".format(maxrecall)
    return fitness_list

def selection(fitness_list,population):
    total = sum(fitness_list)
    prob_list = [f/float(total) for f in fitness_list]
    cumulative_prob_list = []
    cumulative_prob_list.append(prob_list[0])
    for i in range(1,30):
        cumulative_prob_list.append(cumulative_prob_list[i-1]+prob_list[i])
    random_list = np.random.random(30)
    new_list = []
    for i in random_list:
        new_list.append(cumulative_prob_list.index(min(x for x in cumulative_prob_list if x>i)))
    temp = np.zeros((30,44),dtype=np.int)
    for i in range(30):
        #print i,new_list[i]
        temp[i] = population[new_list[i]]
    return temp
    #print population[10]
    #print temp[10]
    #print new_list

def crossover(crossover_rate,population):
    num_chromosomes = int(math.ceil(30*crossover_rate))
    c = np.arange(30)
    np.random.shuffle(c)
    c = c[:num_chromosomes]
    #print c
    for i in range(num_chromosomes):
        crossover_point = int(math.floor(random.uniform(0,1)*44))
        #print "-----------------------"
        #print str(c[i%num_chromosomes]) + "," + str(c[(i+1)%num_chromosomes]) + "," + str(crossover_point)
        #print population[c[i%num_chromosomes]]
        a = population[c[i%num_chromosomes],0:crossover_point]
        b = population[c[(i+1)%num_chromosomes],crossover_point:]
        population[c[i%num_chromosomes]] = np.concatenate([a,b])
        #print population[c[i%num_chromosomes]]
        #print "-----------------------"
        #print crossover_point
        #print str(c[i%num_chromosomes]) + "," + str(c[(i+1)%num_chromosomes])
    #print c
    return population

def mutation(mutation_rate,population):
    #print population.shape[0]*population.shape[1]
    num_genes = int(math.ceil(population.shape[0]*population.shape[1]*mutation_rate))
    pairs = [(i,j) for i in range(30) for j in range(44)]
    random.shuffle(pairs)
    #print pairs
    pairs = np.array(pairs)
    #print pairs
    #print pairs.shape
    pairs = pairs[0:num_genes,:]
    #print pairs.shape
    for i in range(pairs.shape[0]):
        population[pairs[i,0],pairs[i,1]] ^= 1

    return population


df = pd.read_csv('/home/rahul/Desktop/SC/SPECTF_New.csv')
df = df.sample(frac=1)
#print df.head()
A = df.as_matrix()
crossover_rate = 0.25
mutation_rate = 0.1

#only naive bayes
population = np.ones(44,dtype=np.int)
acc,precision,recall = objective_func(A,population)
print "Plain Naive bayes:"
print "Accuracy = {}".format(acc) + " Precision = {}".format(precision) + " Recall = {}".format(recall)

#mutation_rate = 0.1
population = init_population()
max_acc=0
maxprecision=0
maxrecall=0
chromosome=0
#print population[1]
for i in range(10):
    print "--------------------------------------------------------------------------------------------"
    print "GA Iteration: "+str(i+1)
    fitness_list = fitness_func(A,population)
    population = selection(fitness_list,population)
    #print population[1]
    population = crossover(crossover_rate,population)
    #print population[1]
    #print population.shape
    population = mutation(mutation_rate,population)
    print "--------------------------------------------------------------------------------------------"
print "Accuracy for naive bayes with GA : {}".format(max_acc) + " Precision = {}".format(maxprecision) + " Recall = {}".format(maxrecall)
print "Chromosome with highest accuracy = {}".format(chromosome)
#print "-----------------------"
#print "New chromosomes after selection:"
#print population[1]
#print population.shape
#print population.shape
#for i in range(10):
#    train_data,test_data = input_split(X,i)

#    Y = train_data[:,44]
#    YT = test_data[:,44]
#    XT = test_data[:,0:44]
#    X = train_data[:,0:44]

    #naive_bayes(X,Y,YT,XT)


    #   print classlabel



#print float(len(yeslist))/(len(yeslist)+len(nolist))
#print len(meanyes)
#print meanno

#print stdyes
#print stdno

#print X.shape[0]
#print Y
