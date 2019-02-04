import numpy
import solution
import pylab

def draw_decision(X, y, classifier, at1, at2, grid=50):

    points = numpy.take(X, [at1, at2], axis=1)
    maxx, maxy = numpy.max(points, axis=0)
    minx, miny = numpy.min(points, axis=0)
    difx = maxx - minx
    dify = maxy - miny
    maxx += 0.02*difx
    minx -= 0.02*difx
    maxy += 0.02*dify
    miny -= 0.02*dify

    for c,(x,y) in zip(y,points):
        pylab.text(x,y,str(c), ha="center", va="center")
        pylab.scatter([x],[y],c=["b","r"][c!=0], s=200)

    num = grid
    prob = numpy.zeros([num, num])
    for xi,x in enumerate(numpy.linspace(minx, maxx, num=num)):
        for yi,y in enumerate(numpy.linspace(miny, maxy, num=num)):
            #probability of the closest example
            diff = points - numpy.array([x,y])
            dists = (diff[:,0]**2 + diff[:,1]**2)**0.5 #euclidean
            ind = numpy.argsort(dists)
            prob[yi,xi] = classifier(X[ind[0]])[1]

    pylab.imshow(prob, extent=(minx,maxx,maxy,miny))

    pylab.xlim(minx, maxx)
    pylab.ylim(miny, maxy)
    pylab.xlabel(at1)
    pylab.ylabel(at2)

    pylab.show()

X,y = solution.load('reg.data')

learner = solution.LogRegLearner(lambda_=0.)
classifier = learner(X,y)

draw_decision(X, y, classifier, 0, 1)

learner = solution.LogRegLearner(lambda_=0.01)
classifier = learner(X,y)

draw_decision(X, y, classifier, 0, 1)

learner = solution.LogRegLearner(lambda_=0.3)
classifier = learner(X,y)

draw_decision(X, y, classifier, 0, 1)

for lam in [0.0, 0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
    learner = solution.LogRegLearner(lambda_=lam)
    res = solution.test_cv(learner, X, y)
    res1 = solution.test_learning(learner, X, y)
    print("Tocnost cv:", round(solution.CA(y, res), 3))  # argumenta sta pravi razredi, napovedani
    print("Tocnost learning:", round(solution.CA(y, res1), 3))  # argumenta sta pravi razredi, napovedani
    print("\n")

