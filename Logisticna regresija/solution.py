import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import math
from random import shuffle

np.seterr(divide='ignore')
np.seterr(invalid='ignore')

def load(name):
    """ 
    Odpri datoteko. Vrni matriko primerov (stolpci so znacilke) 
    in vektor razredov.
    """
    data = np.loadtxt(name)
    X, y = data[:, :-1], data[:, -1].astype(np.int)
    return X, y


def h(x, theta):
    """ 
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    # ... dopolnite (naloga 1)
    v = 0
    for i in range(len(x)):
        v += theta[i]*x[i]
    return 1/(1+math.e**(-v))


def cost(theta, X, y, lambda_):
    """ 6.4 (6.8) - tm maksimizira, mi mormo minimizirat (- spredi dat)
    Vrednost cenilne funkcije.
    """
    # ... dopolnite (naloga 1, naloga 2)
    # tuki pomojem dimenzije ne stimajo

    n = len(y)
    vr = 0
    for i in range(n):
        hi = h(X[i], theta)
        vr += y[i]*np.log(hi) + (1-y[i])*np.log(1 - hi)
    rez = -vr/n
    regularizacija = (lambda_ / (2 * n)) * sum(theta ** 2)
    return rez + regularizacija


def grad(theta, X, y, lambda_):
    """ 6.5
    Odvod cenilne funkcije. Vrne numpyev vektor v velikosti vektorja theta.
    """
    # ... dopolnite (naloga 1, naloga 2)
    #len(theta)=28 len(X[0])=28
    n = len(X[0])
    m = X.shape[0]
    vek = []
    # print(X[:,1])
    for j in range(n):
        sum = 0
        for i in range(m):
            hi = h(X[i], theta)
            sum += (y[i]-hi)*X[i][j]
        sum = (-sum/m) + ((lambda_ * theta[j]) / m)
        vek.append(sum)
    return np.array(vek)


def num_grad(theta, X, y, lambda_):
    """ v sublajmu
    Odvod cenilne funkcije izracunan numericno.
    Vrne numpyev vektor v velikosti vektorja theta.
    Za racunanje gradienta numericno uporabite funkcijo cost.
    """
    # ... dopolnite (naloga 1, naloga 2)
    # kaj je kle epsilon?

    epsilon = 0.0001
    n = len(X[0])
    vek = []
    for i in range(n):
        t = theta[i]
        t1 = t + epsilon
        theta[i] = t1
        c1 = cost(theta, X, y, lambda_)

        t2 = t - epsilon
        theta[i] = t2
        c2 = cost(theta, X, y, lambda_)
        theta[i] = t
        vek.append((c1-c2)/(2*epsilon))
    return np.array(vek)


class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = np.hstack(([1.], x))
        p1 = h(x, self.th)  # verjetno razreda 1
        return [1-p1, p1]


class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = np.hstack((np.ones((len(X), 1)), X))

        # optimizacija
        theta = fmin_l_bfgs_b(
            cost,
            x0=np.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri učenju.
    To je napačen način ocenjevanja uspešnosti!

    Primer klica:
        res = test_learning(LogRegLearner(lambda_=0.0), X, y)
    """
    c = learner(X,y)
    results = [c(x) for x in X]
    return results


def test_cv(learner, X, y, k=5):
    # ... dopolnite (naloga 3)
    # mormo vedt kje je blo to na zacetku ne mormo kr cele shufflat, ker pol na koncu rabmo drgac
    shuf = [i for i in range(len(X))]
    shuffle(shuf)

    foldi = np.array_split(shuf, k)

    napoved = []
    for i in range(0, k):
        ucniX = []
        ucniY = []
        testni = []
        for j in range(len(foldi)):
            for fix in foldi[j]:
                if j != i:
                    ucniX.append(X[fix])
                    ucniY.append(y[fix])
                else:
                    testni.append(X[fix])

        classifier = learner(np.array(ucniX), ucniY)
        napoved = napoved + [classifier(t) for t in testni]

    rez = []
    for i in range(0, len(napoved)):
        for j in range(0, len(shuf)):
            if shuf[j] == i:
                rez.append(napoved[j])
    return rez

def CA(real, predictions):
    # ... dopolnite (naloga 3)
    stUjemanj = 0
    for i, p in enumerate(predictions):
        p1, p2 = p
        if p2 >= 0.5:
            p2 = 1
        else:
            p2 = 0
        # ce to drzi se ujemata
        if (p2 + real[i]) % 2 == 0:
            stUjemanj += 1
    return stUjemanj / len(predictions)


def AUC(real, predictions):
    # ... dopolnite (dodatna naloga)
    # rabmo par 0-1, ce un k ma 1 vecjo verjetnost prstejemo 1, ce enaka prstejemo 0.5. in delimo s st parov ki smo jih primerjal
    n = len(real)
    print(predictions)
    pred = [p[1] for p in predictions]
    sum = 0
    stPrimerjanj = 0
    for i in range(n):
        for j in range(n):
            # ce je en 0 in en 1
            if j > i and real[i] != real[j]:
                # potem smo primerjali 2 elementa
                stPrimerjanj += 1
                # i-ti = 1 in j-ti = 0
                if real[i] > real[j]:
                    # ce je tudi prediction itega vecji od jtega
                    if pred[i] > pred[j]:
                        sum += 1
                    elif pred[i] == pred[j]:
                        sum += 0.5
                # i-ti = 0 in j-ti = 1
                else:
                    if pred[i] < pred[j]:
                        sum += 1
                    elif pred[i] == pred[j]:
                        sum += 0.5
    return sum/stPrimerjanj


if __name__ == "__main__":
    # Primer uporabe

    X, y = load('reg.data')

    learner = LogRegLearner(lambda_=0.0)
    classifier = learner(X, y)  # dobimo model

    napoved = classifier(X[0])  # napoved za prvi primer
    #napoved[0] = round(napoved[0], 4)
    #napoved[1] = round(napoved[1], 4)
    print(napoved)

