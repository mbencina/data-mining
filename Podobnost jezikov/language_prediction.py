import glob
import math
import os.path
from random import sample
from unidecode import unidecode
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
#import sys
#sys.stdout = open('rez_odstavki.txt','wt')

# FUNKCIJE
def kmers(s, k): # s je beseda iz kjere terko nardimo
	for i in range(len(s)-k+1):
		yield s[i:i+k] # generira izhode

def kosinusDist(dataS, dataC, a, b):
	unija = dataS[a] | dataS[b]
	stevec = 0
	for el in unija:
		aa = dataC[a].get(el) or 0
		bb = dataC[b].get(el) or 0
		stevec+=aa*bb
	sumA=0
	for aa in dataC[a]:
		sumA+=(dataC[a].get(aa))**2
	sumB = 0
	for bb in dataC[b]:
		sumB+=(dataC[b].get(bb))**2
	# manjsi kot je rezultat blizje sta si drzavi, zato 1 - ...
	return 1 - (stevec/(math.sqrt(sumA)*math.sqrt(sumB)))

def kosinusDistOdst(dataS, dataC, a, odstS, odstC, b):
	# uporabljeno v funkciji ugotoviJezik
	presek = dataS[a] & odstS[b]
	stevec = 0
	for el in presek:
		aa = dataC[a].get(el)
		bb = odstC[b].get(el)
		stevec+=aa*bb
	sumA=0
	for aa in dataC[a]:
		sumA+=(dataC[a].get(aa))**2
	sumB = 0
	for bb in odstC[b]:
		sumB+=(odstC[b].get(bb))**2
	# manjsi kot je rezultat blizje sta si drzavi, zato 1 - ...
	return 1 - (stevec/(math.sqrt(sumA)*math.sqrt(sumB)))

# izracun skupne razdaje
def razd (medoidi, plebejci, kosRazd, drzave):
	sum=0
	# ce je drzava medoid je razdalja do nje tako ali tako = 0
	for p in plebejci:
		min = 1
		# iscemo najblizjo drzavo
		for m in medoidi:
			nov = kosRazd[drzave.index(m)][drzave.index(p)]
			if nov < min:
				min = nov
		sum += min
	return sum

def swap (med, pleb, m, p, i, j):
	pleb.remove(p)
	pleb.insert(j, m)
	med.remove(m)
	med.insert(i, p)
	pass

def kMedoids (drz, stMed, kosRazd, drzave):
	# dolocimo random medoide
	medoidi=sample(drzave, stMed)
	# dolocimo se tiste, ki niso medoidi
	plebejci = [d for d in drz if d not in medoidi]
	# izracunamo zacetno razdaljo
	razdalja = razd(medoidi, plebejci, kosRazd, drzave)

	# menjali bomo medoide s plebejci, ce bo cena manjsa
	for i in range(0, len(medoidi)):
		# plebejci so vse tiste drzave, ki niso medoidi
		for j in range(0, len(plebejci)):
			# swap zamenja i-ti element plebejcev in j-ti element medoidov
			swap(medoidi, plebejci, medoidi[i], plebejci[j], i, j)
			# izracunamo razdaljo2 (ce smo swapali elementa)
			razdalja2 = razd(medoidi, plebejci, kosRazd, drzave)
			if razdalja2 < razdalja:
				razdalja = razdalja2
				#break
			else:
				# swapamo nazaj
				swap(medoidi, plebejci, medoidi[i], plebejci[j], i, j)
	# st : razdalja (se printa v mainu, kjer smo 5x klicali to funkcijo)
	return razdalja, medoidi

def matrikaRazdalj (drzave, dataSet, dataCnt):
	n = len(drzave)
	kosRazd = [[None for _ in range(n)] for _ in range(n)]
	for i in range(n):
		for j in range(n):
			kosRazd[i][j] = kosinusDist(dataSet, dataCnt, drzave[i], drzave[j])
	return kosRazd

def ugotoviJezik (dir):
	# funkcija za 3. del naloge, postopek je podoben kot v 1. delu
	corpus = {}
	for file_name in glob.glob(dir):
		# iz mape odstavki preberemo imena datotek
		name = os.path.splitext(os.path.basename(file_name))[0]
		text = " ".join([line.strip() for line in open(file_name, "rt", encoding="utf8").readlines()])
		text = text.lower()
		corpus[name] = unidecode(text)

	odstSet = {k: set(kmers(corpus[k], 3)) for k in corpus}
	odstCnt = {k: dict(Counter(kmers(corpus[k], 3))) for k in corpus}
	odstavki = list(odstSet.keys())

	# iscemo 3 najbolj verjetne jezike, ki jih bomo spravili v list "jezik"
	for o in odstavki:
		jezik = [None, None, None]
		min = [1, 1, 1]
		n = len(min)
		for d in drzave:
			nov = kosinusDistOdst(dataSet, dataCnt, d, odstSet, odstCnt, o)
			for i in range(0, n):
				if nov < min[i]:
					min.insert(i, nov)
					min.pop()
					jezik.insert(i, d)
					jezik.pop()
					break
		print(o, "je napisan v jeziku:")
		for i in range(0, n):
			print(jezik[i], "z verjetnostjo", round(1 - min[i], 2))
		print("")
	pass

def defClustri (medoidi, drzave):
	# vrnemo clustre, na podalgi podanih medoido in drzav
	rez = []
	# naredimo prazne clustre
	for i in range(len(medoidi)):
		rez.append([])
	for d in drzave:
		min = 1
		medoid = -1
		# iscemo najblizjega medoida, in drzavo damo v cluster
		for i, m in enumerate(medoidi):
			nov = kosRazd[drzave.index(m)][drzave.index(d)]
			if nov < min:
				min = nov
				medoid = i
		rez[medoid].append(d)
	return rez

def distToClust (drz, clust, drzave, kosRazd):
	# racunanje razdalj vseh drzav znotraj clustra
	sum = 0
	for c in clust:
		sum += kosRazd[drzave.index(drz)][drzave.index(c)]
	return sum / len(clust)

def silhueta (drzave, clustri, kosRazd):
	# funkcija za izracun silhuete
	silh = []
	for i, d in enumerate(drzave):
		ai = None
		# obcutno prevelika stevilka za razdalje
		bi = 9999
		# racunano po formuli na Wiki
		for c in clustri:
			# ce je to cluster od drzave
			if d in c:
				ai = distToClust(d, c, drzave, kosRazd)
			else:
				tmp = distToClust(d, c, drzave, kosRazd)
				# ce smo nasli najblizji cluster od ostalih, ki drzave ne vsebujejo
				if tmp < bi:
					bi = tmp
		silh.append((bi - ai) / max(bi, ai))
	return silh

def izrisiHistogram (drzave, clustri, silhueta):
	izris = []
	izrisLen = []
	# priprava za risanje
	for i, c in enumerate(clustri):
		izris.append([])
		for j, dr in enumerate(c):
			izris[i].append(silhueta[drzave.index(dr)])
		izris[i].sort(reverse=True)
		izrisLen.append(len(izris[i]))

	izrisList = []
	for c in izris:
		for dr in c:
			izrisList.append(dr)

	# risanje crt
	plt.axis([0, 1, 0.5, 20.5])
	barve = ['y', 'r', 'c', 'b', 'g']
	plt.xlabel('Silhueta')
	plt.ylabel('Jeziki')
	cnt = izrisLen[0]
	ix = 0
	for i in range(len(drzave)):
		plt.plot([0, izrisList[i]], [len(drzave) - i, len(drzave) - i], barve[ix], linewidth=10)
		cnt -= 1
		if cnt == 0:
			ix += 1
			ix %= 5
			cnt = izrisLen[ix]

	plt.show()
	pass

if __name__ == "__main__":
	# 1. del naloge
	corpus = {}
	for file_name in glob.glob("drzave/*"):
		text = " ".join([line.strip() for line in open(file_name, "rt", encoding="utf8").readlines()])
		text = text.lower()
		name = os.path.splitext(os.path.basename(file_name))[0]
		corpus[name] = unidecode(text)

	dataSet = {k: set(kmers(corpus[k], 3)) for k in corpus}
	dataCnt = {k: dict(Counter(kmers(corpus[k], 3))) for k in corpus}
	drzave = list(dataSet.keys())

	# matrika kosinusnih razdalj
	kosRazd = matrikaRazdalj(drzave, dataSet, dataCnt)

	# funkcija za izračun medoidov:
	# kMedoids(drzave, 5, kosRazd, drzave)

	# 2. del naloge
	clustri = []

	stIter = 100
	najboljsaSilh = [] # vec je bolje
	najboljsiClustri = None
	vrNajb = 0
	najslabsaSilh = []
	najslabsiClustri = None
	vrNajsl = 9999
	for i in range(0, stIter):
		r = kMedoids(drzave, 5, kosRazd, drzave)
		# print(i, ": ", round(r[0], 2))
		medoidi = r[1]
		clustri = defClustri(medoidi, drzave)
		silh = silhueta(drzave, clustri, kosRazd)
		vrSilh = sum(silh)
		if vrNajb < vrSilh:
			najboljsaSilh = silh
			vrNajb = sum(najboljsaSilh)
			najboljsiClustri = clustri
		if vrNajsl > vrSilh:
			najslabsaSilh = silh
			vrNajsl = sum(najslabsaSilh)
			najslabsiClustri = clustri

	#print("Vrednost najboljse silhuete:", vrNajb)
	#print("Vrednost najslabse silhuete:", vrNajsl)
	#print(najboljsiClustri)
	#print(najslabsiClustri)

	# izris grafa
	izrisiHistogram(drzave, najboljsiClustri, najboljsaSilh)
	izrisiHistogram(drzave, najslabsiClustri, najslabsaSilh)

	# 3. del naloge - odkumentiraj funkcijo ugotoviJezik za izpis rezultata
	ugotoviJezik("odstavki/*")

"""
slv - slovenia
src3 - serbia
src1 - bosnia
rus - russia
mkj - macedonian

ger - germany
eng - english
dut - dutch
nrr - norwegian

frn - french
spn - spanish
itn - italian
rum - romanian
por - portuguese

hng - hungarian
grk - greek
pcm - nigerian
chn - chinese
jpn - japanese
swa - swahili
"""
