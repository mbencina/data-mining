import math
import csv

# uporabljeno za izpis v drevo.txt
# import sys
# sys.stdout = open('test.txt','wt')

def read_file(file_name):
    """
    Read and process data to be used for clustering.
    :param file_name: name of the file containing the data
    :return: dictionary with element names as keys and feature vectors as values
    """
    drzave = []
    data = {}

    f = open(file_name, "rt", encoding="latin1")
    # l je vrstica v f, i je st vrstice
    for i, l in enumerate(csv.reader(f)):
        if i > 0:
            # drzava, ki je napisana v vrstici levo
            glasovana = l[1]
            # dodajamo tocke, ki jih je dobila glasovana drzava posamezno leto
            vrsta = [float(e) if e else None for e in l[16:63]]
            # namesto, da drzava zase ne more glasovati, sebi da 12 tock (da bolje ocenimo blizino)
            for i in range(0, len(vrsta)):
                if drzave[i]!=glasovana:
                    data[drzave[i]].append(vrsta[i])
                else:
                    data[drzave[i]].append(float(12))
        else:
            # v tem primeru beremo drzave, ki jih shranimo
            drzave = [e.strip() for e in l[16:63]]  # drzave so od 16-63
            for drzava in drzave:
                data[drzava] = []

    return data



class HierarchicalClustering:
    def __init__(self, data):
        """Initialize the clustering"""
        self.data = data
        # self.clusters stores current clustering. It starts as a list of lists
        # of single elements, but then evolves into clusterings of the type
        # [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
        self.clusters = [[name] for name in self.data.keys()]

    def row_distance(self, r1, r2):
        """
        Distance between two rows.
        Implement either Euclidean or Manhattan distance.
        Example call: self.row_distance("Polona", "Rajko")
        """
        x = self.data[r1]
        y = self.data[r2]
        # izracunamo razdaljo
        dist = math.sqrt(sum([(a - b) ** 2  for a, b in zip(x, y) if not a is None and not b is None])) #skipa none
        if dist>0:
            return dist
        else:
            # v tem primeru drzavi nista bili istocasno na evroviziji - returnamo neko povprecno vrednost, ki je okoli 55
            # izracunano z metodo avg_clusters()
            return float(55)

    def avg_clusters(self):
        # funkcija je bila uporabljena za izracun povprecne razdalje med dvojico vseh drzav
        # ob klicu run() se funkcija ne klice
        sum=0
        kol=0
        for i in range(0, len(self.clusters)-1):
            for j in range(i+1, len(self.clusters)):
                sum+=self.cluster_distance(self.clusters[i], self.clusters[j])
                kol+=1
        return sum/kol

    def cluster_names(self, c, arr):
        # funkcija nam s pomocjo rekurzije vrne array stringov v clustru - vse na istem nivoju za lazjo obdelavo
        if (isinstance(c, str)):
            return arr.append(c)
        for i, cn in enumerate(c):
            self.cluster_names(c[i], arr)
        return arr


    def cluster_distance(self, c1, c2):
        """ vec k so si dal pik bl so si bliz kao
        Compute distance between two clusters.
        Implement either single, complete, or average linkage.
        Example call: self.cluster_distance(
            [[["Albert"], ["Branka"]], ["Cene"]],
            [["Nika"], ["Polona"]])
        """
        # taktika,da izberemo najbolj oddaljenega iz clustra
        # kako dobis posemezen ime ven? - z rekurzijo
        arr1 = self.cluster_names(c1, [])
        arr2 = self.cluster_names(c2, [])
        """ taktika povprecja se ni najbolje obnesla
        sum=0
        for a1 in arr1:
            for a2 in arr2:
                sum+=self.row_distance(a1, a2)
        kolicnik=len(arr1)*len(arr2)
        # vrnemo utezeno povprecje
        return float(sum/kolicnik)
        """
        max=0
        for a1 in arr1:
            for a2 in arr2:
                nov = self.row_distance(a1, a2)
                if nov>max:
                    max=nov
        return max

    def closest_clusters(self):
        """
        Find a pair of closest clusters and returns the pair of clusters and
        their distance.

        Example call: self.closest_clusters(self.clusters)
        """
        minDist=9999999
        r1=None
        r2=None
        for i in range(0, len(self.clusters)-1):
            for j in range(i+1, len(self.clusters)):
                newDist=self.cluster_distance(self.clusters[i], self.clusters[j])
                if (newDist<minDist):
                    r1=self.clusters[i]
                    r2=self.clusters[j]
                    minDist=newDist
        return [minDist, r1, r2]

    def cluster_avg_distance(self, c1, c2):
        # funkcija uporabljena za izracun preferiranih drzav
        arr1 = self.cluster_names(c1, [])
        arr2 = self.cluster_names(c2, [])
        sum=0
        for a1 in arr1:
            for a2 in arr2:
                sum+=self.row_distance(a1, a2)
        kolicnik=len(arr1)*len(arr2)
        # vrnemo utezeno povprecje
        return float(sum/kolicnik)

    def run(self):
        """
        Given the data in self.data, performs hierarchical clustering.
        Can use a while loop, iteratively modify self.clusters and store
        information on which clusters were merged and what was the distance.
        Store this later information into a suitable structure to be used
        for plotting of the hierarchical clustering.
        """

        while (len(self.clusters) > 2):
            c = self.closest_clusters()
            i1 = self.clusters.index(c[1])
            i2 = self.clusters.index(c[2])
            # elementa zdruzimo
            self.clusters[i1] = [c[1], c[2]]
            del (self.clusters[i2])

        pass

    def tree_rek(self, c, glob):
        # funkcija ki ji klice plot_tree() za izris drevesa
        if isinstance(c[0], str):
            print(glob + "---- " + c[0])
            return
        else:
            self.tree_rek(c[0], glob+"    ")

        print(glob + "----|")

        if isinstance(c[1], str):
            print(glob + "---- " + c[1])
            return
        else:
            self.tree_rek(c[1], glob+"    ")
        pass

    def plot_tree(self):
        """
        Use cluster information to plot an ASCII representation of the cluster
        tree.
        """
        print("\n")
        self.tree_rek(self.clusters, "")
        pass


if __name__ == "__main__":
    DATA_FILE = "eurovision-data.csv"
    hc = HierarchicalClustering(read_file(DATA_FILE))
    hc.run()
    hc.plot_tree()

