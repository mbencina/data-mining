from collections import defaultdict
import numpy
from scipy.optimize import fmin_l_bfgs_b
import csv
import scipy
import scipy.sparse
from scipy.sparse import csr_matrix
from datetime import datetime
import time

import sys
sys.stdout = open('napoved.txt', 'wt')

def read_file(file_name):
    podatki = []
    data = {}

    f = open(file_name, "rt", encoding="utf8")
    # l je vrstica v f, i je st vrstice
    for i, l in enumerate(csv.reader(f)):
        if i > 0:
            vrsta = l[0].split('\t')
            for j in range(0, len(vrsta)):
                data[podatki[j]].append(vrsta[j])
        else:
            # prva vrsta
            podatki = l[0].split('\t')
            for p in podatki:
                data[p] = []
    return data

def read_vreme(file_name):
    podatki = []
    data = {}

    f = open(file_name, "rt", encoding="utf8")
    # l je vrstica v f, i je st vrstice
    for i, l in enumerate(csv.reader(f)):
        if i > 0:
            for j, ksz in enumerate(podatki):
                data[podatki[j]].append(l[j])
        else:
            # prva vrsta
            # dodajam jim stevilke, ker imajo stolpci drugace isti key
            for i, p in enumerate(l):
                data[p + str(i)] = []
                podatki.append(p + str(i))

    return data

def toSeconds(s): # iz stringa v sekunde
    # najprej damo v pravi fomat
    tmp = datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f')
    # nato damo v sekunde
    return time.mktime(tmp.timetuple())

def toDate(i): # iz sekund to date
    return datetime.utcfromtimestamp(i)
# thete s fitom napovemo

def secToHr(i):
    return (i/3600)%24

def zgradiModel (pod, ix1, ix2, lin, linSt, avg):
    for i in range(len(pod["Arrival time"])):
        if ix1 <= i  and  i < ix2:
            continue
        dTime = pod["Departure time"][i]
        aTime = pod["Arrival time"][i]
        stBusa = pod["Route Direction"][i]
        if stBusa not in lin.keys():
            lin[stBusa] = [0] * 24
            linSt[stBusa] = [0] * 24
        # v sekunde odstet in pol razliko dat v casi. pristejemo 3600, ker pretvorba vzame 1 uro
        dTime = toSeconds(dTime) + 3600.001
        aTime = toSeconds(aTime) + 3600.001
        # racunanje povprecja po urah
        lin[stBusa][round(secToHr(dTime)) % 24] += (aTime-dTime)
        linSt[stBusa][round(secToHr(dTime)) % 24] += 1

    for k in lin.keys():
        if (sum(linSt[k]) == 0):
            povpr = avg  # vtemtakem nimamo ustrezniih podatkov in nastavimo na avg
        else:
            povpr = sum(lin[k]) / sum(linSt[k])
        for i, st in enumerate(lin[k]):
            if linSt[k][i] == 0:
                # ce slucjano hocemo cas, ki ni bil definiran na testnih podatkih
                lin[k][i] = povpr
            else:
                lin[k][i] = lin[k][i]/linSt[k][i]
    return lin

# X so pod
def crossvalidate(pod, y, ln, lnSt, avg):
    n = len(y)
    # 10-fold
    sum = 0
    sumLen = 0
    for i in range(10):
        # nastavimo prvi in drugi index, med indeksoma so testni podatki
        ix1 = int(i*(n/10))
        ix2 = int((i+1)*(n/10))
        # ponastavimo vrednosti
        for k in ln:
            ln[k] = [0] * 24
            lnSt[k] = [0] * 24
        # zgradili bomo model nad 9/10 podatkov
        napoved = zgradiModel(pod, ix1, ix2, ln, lnSt, avg)

        #testiranje modela na 1/9 podatkov (fejk testni)
        for i in range(ix1, ix2):
            dTime = pod["Departure time"][i]
            # pretvorba vzame 1h zato jo damo nazaj, +0.001 se doda zaradi formata izpisa
            dTime = toSeconds(dTime) + 3600.001
            napovedaniCas = napoved[pod["Route Direction"][i]][round(secToHr(dTime)) % 24] + dTime
            sum += abs(y[i] - napovedaniCas)
            sumLen += 1
    # in se popucamo za sabo
    for k in ln:
        ln[k] = [0] * 24
        lnSt[k] = [0] * 24
    return sum / sumLen

if __name__ == "__main__":

    # dodatno: uporabimo vreme
    vreme = read_vreme("ljvreme_2012.csv")
    padavine = {}
    padavine["dan"] = []
    padavine["sneg"] = []
    padavine["padavine"] = []

    for i in range(len(vreme["dan0"])):
        padavine["dan"].append(vreme["dan0"][i])
        padavine["sneg"].append(vreme["sneg21"][i])
        padavine["padavine"].append(vreme["padavine20"][i])

    # obdelava podatkov
    pod = read_file("train.csv")
    y = []
    avgUre = [0] * 24
    avgUreSt = [0] * 24
    linije = {}
    linijeSt = {}
    n = 0
    avg = 0
    novPod = {}
    for k in pod.keys():
        novPod[k] = []
    for i in range(len(pod["Arrival time"])):
        dTime = pod["Departure time"][i]
        # po veliko testih sta se najbolje obnesla november in februar, ker sta najbolj reprezentativna. ostale izlocimo
        if toDate(toSeconds(dTime)).month == 11 or toDate(toSeconds(dTime)).month == 2:
            # te podatke bomo uporabili
            stBusa = pod["Route Direction"][i]
            novPod["Departure time"].append(pod["Departure time"][i])
            novPod["Arrival time"].append(pod["Arrival time"][i])
            novPod["Route Direction"].append(stBusa)
            if stBusa not in linije.keys():
                linije[stBusa] = [0] * 24
                linijeSt[stBusa] = [0] * 24
            # konverzija v sekunde se zamakne za 1h -> 3600s
            y.append(toSeconds(pod["Arrival time"][i]) + 3600.001)
            # racunanje povprecja za tiste, ki nimajo ustreznih podatkov
            avg += (toSeconds(pod["Arrival time"][i]) - toSeconds(pod["Departure time"][i]))
            n += 1

    avg = avg/n
    # crossvalidacija
    print(crossvalidate(novPod, y, linije, linijeSt, avg))

    #zdaj pa zgradimo dejanski model
    linije = zgradiModel(novPod, -1, -1, linije, linijeSt, avg)

    pod2 = read_file("test.csv")
    for i in range(len(pod2["Arrival time"])):
        dTime = pod2["Departure time"][i]
        dTime = toSeconds(dTime) + 3600.001
        dan = int(toDate(toSeconds(pod["Departure time"][0])).day)  # to je za vreme
        print(toDate(linije[pod2["Route Direction"][i]][round(secToHr(dTime)) % 24] + dTime + int(padavine["sneg"][dan]) + int(padavine["padavine"][dan])))  # izpis
