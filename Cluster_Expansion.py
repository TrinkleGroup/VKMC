from onsager import cluster
import numpy as np
import collections
import itertools
import Transitions
import time
from tqdm import tqdm

class ClusterSpecies():

    def __init__(self, specList, siteList, zero=True):
        """
        Creation to represent clusters from site Lists and species lists
        :param specList: Species lists
        :param siteList: Sites which the species occupy. Must be ClusterSite object from Onsager (D.R. Trinkle)
        :param zero: Whether to make the centroid of the sites at zero or not.
        """
        if len(specList)!= len(siteList):
            raise ValueError("Species and site lists must have same length")
        if not all(isinstance(site, cluster.ClusterSite) for site in siteList):
            raise TypeError("The sites must be entered as clusterSite object instances")
        # Form (site, species) set
        # Calculate the translation to bring center of the sites to the origin unit cell
        self.zero=zero
        self.specList = specList
        self.siteList = siteList
        if zero:
            Rtrans = sum([site.R for site in siteList])//len(siteList)
            self.transPairs = [(site-Rtrans, spec) for site, spec in zip(siteList, specList)]
        else:
            self.transPairs = [(site, spec) for site, spec in zip(siteList, specList)]
        # self.transPairs = sorted(self.transPairs, key=lambda x: x[1])
        self.SiteSpecs = sorted(self.transPairs, key=lambda s: np.linalg.norm(s[0].R))

        hashval = 0
        for site, spec in self.transPairs:
            hashval ^= hash((site, spec))
        self.__hashcache__ = hashval

    def __eq__(self, other):

        if set(self.SiteSpecs) == set(other.SiteSpecs):
            return True
        return False

    def __hash__(self):
        return self.__hashcache__

    def g(self, crys, gop, zero=True):
        specList = [spec for site, spec in self.SiteSpecs]
        siteList = [site.g(crys, gop) for site, spec in self.SiteSpecs]
        return self.__class__(specList, siteList, zero=zero)

    @staticmethod
    def inSuperCell(SpCl, N_units):
        siteList = []
        specList = []
        for site, spec in SpCl.SiteSpecs:
            Rnew = site.R % N_units
            siteNew = cluster.ClusterSite(ci=site.ci, R=Rnew)
            siteList.append(siteNew)
            specList.append(spec)
        return SpCl.__class__(specList, siteList, zero=SpCl.zero)


    def strRep(self):
        str= ""
        for site, spec in self.SiteSpecs:
            str += "Spec:{}, site:{},{} ".format(spec, site.ci, site.R)
        return str

    def __repr__(self):
        return self.strRep()

    def __str__(self):
        return self.strRep()


class VectorClusterExpansion(object):
    """
    class to expand velocities and rates in vector cluster functions.
    """
    def __init__(self, sup, clusexp, Tclusexp, jumpnetwork, NSpec, vacSite, maxorder, maxorderTrans,
                 zeroClusts=True, OrigVac=False):
        """
        :param sup : clusterSupercell object
        :param clusexp: cluster expansion about a single unit cell.
        :param Tclusexp: Transition state cluster expansion
        :param jumpnetwork: the single vacancy jump network in the lattice used to construct sup
        :param NSpec: no. of species to consider (including vacancies)
        :param vacSite: Index of the vacancy site (used in MC sampling and JIT construction)
        :param vacSite: the site of the vacancy as a clusterSite object. This does not change during the simulation.
        :param maxorder: the maximum order of a cluster in clusexp.
        :param maxorderTrans: the maximum order of a transition state cluster
        :param zeroClusts: Same as parameter "zero" of ClusterSpecies class - whether to bring a cluster's centroid to zero or not.
        :param OrigVac: only vacancy-atom pairs with the vacancy at the centre will be considered. This will not use clusexp.
        In this type of simulations, we consider a solid with a single wyckoff set on which atoms are arranged.
        """
        self.chem = 0  # we'll work with a monoatomic basis
        self.sup = sup
        self.N_units = sup.superlatt[0, 0]
        self.Nsites = len(self.sup.mobilepos)
        self.crys = self.sup.crys
        # vacInd will always be the initial state in the transitions that we consider.
        self.clusexp = clusexp
        self.Tclusexp = Tclusexp
        self.maxOrder = maxorder
        self.vacSpec = NSpec - 1
        self.Nvac = 1
        self.NSpec = NSpec
        self.mobList = list(range(NSpec))
        self.vacSite = vacSite  # This stays fixed throughout the simulation, so makes sense to store it.
        self.jumpnetwork = jumpnetwork
        self.zeroClusts = zeroClusts
        self.OrigVac = OrigVac

        start = time.time()
        if OrigVac:
            self.SpecClusters, self.SiteSpecInteractIds, self.InteractionIdDict,\
            self.clust2InteractId, self.maxinteractions = self.InteractsOrigVac()
        else:
            self.SpecClusters = self.recalcClusters()
            self.IndexClusters()  # assign integer integer IDs to each cluster
            end1 = time.time()
            print("\t built {} clusters:{:.4f} seconds".format(len([cl for clist in self.SpecClusters for cl in clist])
                                                               , end1 - start))
            self.SiteSpecInteractIds, self.InteractionIdDict,\
            self.clust2InteractId, self.maxinteractions = self.generateSiteSpecInteracts()
            print("\t built interactions:{:.4f} seconds".format(time.time() - end1))
            # add a small check here - maybe we'll remove this later

        print("Built Species Clusters : {:.4f} seconds".format(time.time() - start))

        start = time.time()
        self.vecClus, self.vecVec, self.clus2LenVecClus = self.genVecClustBasis(self.SpecClusters)
        print("Built vector bases for clusters : {:.4f}".format(time.time() - start))

        start = time.time()
        self.KRAexpander = Transitions.KRAExpand(sup, self.chem, jumpnetwork, maxorderTrans, Tclusexp, NSpec, self.Nvac, vacSite)
        print("Built KRA expander : {:.4f}".format(time.time() - start))

        start = time.time()
        self.indexVclus2Clus()  # Index vector cluster list to cluster symmetry groups
        self.indexClustertoVecClus()  # Index where in the vector cluster list a cluster is present
        self.indexClustertoSpecClus()  # Index clusters to symmetry groups
        print("Built Indexing : {:.4f}".format(time.time() - start))

    def recalcClusters(self):
        """
        Intended to take in a site based cluster expansion and recalculate the clusters with species in them
        """
        allClusts = set()
        symClusterList = []
        self.SpecClust2Clus = {}
        for clSetInd, clSet in enumerate(self.clusexp):
            for clust in list(clSet):
                Nsites = len(clust.sites)
                occupancies = list(itertools.product(self.mobList, repeat=Nsites))
                for siteOcc in occupancies:
                    # Make the cluster site object
                    if self.OrigVac:
                        if siteOcc[0] != self.vacSpec:
                            continue
                    ClustSpec = ClusterSpecies.inSuperCell(ClusterSpecies(siteOcc, clust.sites, zero=self.zeroClusts), self.N_units)
                    # check if this has already been considered
                    if ClustSpec in allClusts:
                        continue
                    # Check if number of each species in the cluster is okay
                    mobcount = collections.Counter(siteOcc)
                    # Check if the number of vacancies is kept to the allowed number
                    if mobcount[self.vacSpec] > self.Nvac:
                        continue
                    # Otherwise, find all symmetry-grouped counterparts
                    if self.OrigVac:
                        newSymSet = set([ClusterSpecies.inSuperCell(ClustSpec.g(self.crys, g, zero=self.zeroClusts), self.N_units)
                                         for g in self.crys.G])
                    else:
                        newSymSet = set([ClustSpec.g(self.crys, g, zero=self.zeroClusts) for g in self.crys.G])

                    allClusts.update(newSymSet)
                    newList = list(newSymSet)
                    self.SpecClust2Clus[len(symClusterList)] = clSetInd
                    symClusterList.append(newList)

        return sorted(symClusterList, key=lambda sList:np.linalg.norm(sList[0].SiteSpecs[-1][0].R))

    def genVecClustBasis(self, specClusters):

        vecClustList = []
        vecVecList = []
        clus2LenVecClus = np.zeros(len(specClusters), dtype=int)
        for clListInd, clList in enumerate(specClusters):
            cl0 = clList[0]
            glist0 = []
            if not self.OrigVac:
                for g in self.crys.G:
                    if cl0.g(self.crys, g, zero=self.zeroClusts) == cl0:
                        glist0.append(g)
            else:
                for g in self.crys.G:
                    if ClusterSpecies.inSuperCell(cl0.g(self.crys, g, zero=self.zeroClusts), self.N_units) == cl0:
                        glist0.append(g)

            G0 = sum([g.cartrot for g in glist0])/len(glist0)
            vals, vecs = np.linalg.eig(G0)
            vecs = np.real(vecs)
            vlist = [vecs[:, i]/np.linalg.norm(vecs[:, i]) for i in range(3) if np.isclose(vals[i], 1.0)]
            clus2LenVecClus[clListInd] = len(vlist)

            if clus2LenVecClus[clListInd] == 0:  # If the vector basis is empty, don't consider the cluster
                # vecClustList.append(clList)
                # vecVecList.append([np.zeros(3) for i in range(len(clList))])
                continue

            for v in vlist:
                newClustList = [cl0]
                # The first cluster being the same helps in indexing
                newVecList = [v]
                for g in self.crys.G:
                    if not self.OrigVac:
                        cl1 = cl0.g(self.crys, g, zero=self.zeroClusts)
                    else:
                        cl1 = ClusterSpecies.inSuperCell(cl0.g(self.crys, g, zero=self.zeroClusts), self.N_units)
                    if cl1 in newClustList:
                        continue
                    newClustList.append(cl1)
                    newVecList.append(np.dot(g.cartrot, v))

                vecClustList.append(newClustList)
                vecVecList.append(newVecList)

        return vecClustList, vecVecList, clus2LenVecClus

    def indexVclus2Clus(self):

        self.Vclus2Clus = np.zeros(len(self.vecClus), dtype=int)
        self.Clus2VClus = collections.defaultdict(list)
        for cLlistInd, clList in enumerate(self.SpecClusters):
            if self.clus2LenVecClus[cLlistInd] == 0:  # If the vector basis is empty, don't consider the cluster
                self.Clus2VClus[cLlistInd] = []
                continue
            cl0 = clList[0]
            for vClusListInd, vClusList in enumerate(self.vecClus):
                clVec0 = vClusList[0]
                if clVec0 == cl0:
                    self.Vclus2Clus[vClusListInd] = cLlistInd
                    self.Clus2VClus[cLlistInd].append(vClusListInd)

        self.Clus2VClus.default_factory = None

    def indexClustertoVecClus(self):
        """
        For a given cluster, store which vector cluster it belongs to
        """
        self.clust2vecClus = collections.defaultdict(list)
        for clListInd, clList in enumerate(self.SpecClusters):
            if self.clus2LenVecClus[clListInd] == 0:  # If the vector basis is empty, don't consider the cluster
                continue
            vecClusIndList = self.Clus2VClus[clListInd]
            for clust1 in clList:
                for vecClusInd in vecClusIndList:
                    for clust2Ind, clust2 in enumerate(self.vecClus[vecClusInd]):
                        if clust1 == clust2:
                            self.clust2vecClus[clust1].append((vecClusInd, clust2Ind))

        self.clust2vecClus.default_factory = None

    def indexClustertoSpecClus(self):
        """
        For a given cluster, store which vector cluster it belongs to
        """
        self.clust2SpecClus = {}
        for clListInd, clList in enumerate(self.SpecClusters):
            for clustInd, clust in enumerate(clList):
                self.clust2SpecClus[clust] = (clListInd, clustInd)

    def InteractsOrigVac(self):
        """
        NOTE : only works for monoatomic lattices for now
        """
        allClusts = set()
        symClusterList = []
        siteA = cluster.ClusterSite(ci=(0, 0), R=np.zeros(self.crys.dim, dtype=int))
        assert siteA == self.vacSite

        for siteInd in range(self.Nsites):
            ciSite, RSite = self.sup.ciR(siteInd)
            clSite = cluster.ClusterSite(ci=ciSite, R=RSite)
            if clSite == siteA:
                continue
            for spec in range(self.NSpec-1):
                siteList = [siteA, clSite]
                specList = [self.NSpec - 1, spec]
                SpCl = ClusterSpecies(specList, siteList, zero=self.zeroClusts)
                if SpCl in allClusts:
                    continue
                # Apply group operations
                newsymset = set([ClusterSpecies.inSuperCell(SpCl.g(self.crys, g, zero=self.zeroClusts), self.N_units)
                             for g in self.crys.G])
                allClusts.update(newsymset)
                symClusterList.append(list(newsymset))

        allSpCl = [cl for clSet in symClusterList for cl in clSet]

        # index the clusters
        clust2InteractId = collections.defaultdict(list)
        InteractionIdDict = {}
        for i, SpCl in enumerate(allSpCl):
            self.Clus2Num[SpCl] = i
            clust2InteractId[SpCl].append(i)
            self.Num2Clus[i] = SpCl
            InteractionIdDict[i] = tuple(sorted([(self.sup.index(st.R, st.ci)[0], spec)
                                                 for st, spec in SpCl.SiteSpecs],
                                                key=lambda x: x[0]))

        SiteSpecinteractIds = collections.defaultdict(list)
        for clSet in symClusterList:
            for cl in clSet:
                Id = self.Clus2Num[cl]
                site = cl.siteList[1]
                siteInd = self.sup.index(ci=site.ci, R=site.R)
                spec = cl.specList[1]
                SiteSpecinteractIds[(siteInd, spec)].append(Id)

        SiteSpecinteractIds.default_factory = None
        maxinteractions = max([len(lst) for key, lst in SiteSpecinteractIds.items()])

        return symClusterList, SiteSpecinteractIds, InteractionIdDict, clust2InteractId, maxinteractions

    def generateSiteSpecInteracts(self):
        """
        generate interactions for every site - for MC moves
        """
        allLatTransTuples = [self.sup.ciR(siteInd) for siteInd in range(self.Nsites)]
        InteractionIdDict = {}
        SiteSpecInteractIds = collections.defaultdict(list)
        clust2InteractId = collections.defaultdict(list)

        count = 0
        # Traverse through all the unit cells in the supercell
        for translateInd in tqdm(range(self.Nsites), position=0, leave=True):
            # Now, go through all the clusters and translate by each lattice translation
            for clID, cl in self.Num2Clus.items():
                # get the cluster site
                R = allLatTransTuples[translateInd][1]
                # translate all sites with this translation
                interactSupInd = tuple(sorted([(self.sup.index(st.R + R, st.ci)[0], spec)
                                               for st, spec in cl.SiteSpecs],
                                              key=lambda x: x[0]))
                if interactSupInd in InteractionIdDict:
                    raise ValueError("Interaction encountered twice while either translating same cluster differently"
                                     "or different clusters.")
                # give the new interaction an Index
                InteractionIdDict[count] = interactSupInd

                # For every rep cluster, store which interactions they produce
                clust2InteractId[clID].append(count)

                # For every site and species, store which interaction they belong to
                for siteInd, spec in interactSupInd:
                    SiteSpecInteractIds[(siteInd, spec)].append(count)

                # Increment the index
                count += 1

        maxinteractions = max([len(lst) for key, lst in SiteSpecInteractIds.items()])

        SiteSpecInteractIds.default_factory = None
        clust2InteractId.default_factory = None

        return SiteSpecInteractIds, InteractionIdDict, clust2InteractId, maxinteractions

    def IndexClusters(self):
        """
        Assign a unique integer to each representative cluster. To help identifying them in JIT operations
        """
        allSpCl = [SpCl for SpClList in self.SpecClusters for SpCl in SpClList]
        self.Clus2Num = {}
        self.Num2Clus = {}

        for i, SpCl in enumerate(allSpCl):
            self.Clus2Num[SpCl] = i
            self.Num2Clus[i] = SpCl

    def makeJitInteractionsData(self, Energies):
        """
        Function to represent all the data structures in the form of numpy arrays so that they can be accelerated with
        numba's jit compilations.
        Data structures to cast into numpy arrays:
        SiteInteractions
        KRAexpander.clusterSpeciesJumps - these correspond to transitions - We'll proceed with this later on
        """

        # first, we assign unique integers to interactions
        start = time.time()
        # while we're at it, let's also store which siteSpec contains which interact
        numInteractsSiteSpec = np.zeros((self.Nsites, self.NSpec), dtype=int)
        SiteSpecInterArray = np.full((self.Nsites, self.NSpec, self.maxinteractions), -1, dtype=int)

        for key, interactIdList in self.SiteSpecInteractIds.items():
            keySite = key[0]  # the "index" function applies PBC to sites outside sup.
            keySpec = key[1]
            numInteractsSiteSpec[keySite, keySpec] = len(interactIdList)
            for interactNum, interactId in enumerate(interactIdList):
                SiteSpecInterArray[keySite, keySpec, interactNum] = interactId

        print("Done Indexing interactions : {}".format(time.time() - start))
        # Now that we have integers assigned to all the interactions, let's store their data as numpy arrays
        numInteracts = len(self.InteractionIdDict)

        # 1. Store chemical data
        start = time.time()
        # we'll need the number of sites in each interaction
        numSitesInteracts = np.zeros(numInteracts, dtype=int)

        # and the supercell sites in each interaction
        SupSitesInteracts = np.full((numInteracts, self.maxOrder), -1, dtype=int)

        # and the species on the supercell sites in each interaction
        SpecOnInteractSites = np.full((numInteracts, self.maxOrder), -1, dtype=int)

        for (key, interaction) in self.InteractionIdDict.items():
            numSitesInteracts[key] = len(interaction)
            for idx, (interactSite, interactSpec) in enumerate(interaction):
                SupSitesInteracts[key, idx] = interactSite
                SpecOnInteractSites[key, idx] = interactSpec

        print("Done with chemical data of interactions : {}".format(time.time() - start))

        # 2. Store energy data and vector data
        start = time.time()
        Interaction2En = np.zeros(numInteracts, dtype=float)
        numVecsInteracts = np.full(numInteracts, -1, dtype=int)
        VecsInteracts = np.zeros((numInteracts, 3, 3))
        VecGroupInteracts = np.full((numInteracts, 3), -1, dtype=int)
        for repClusInd, interactionList in self.clust2InteractId.items():
            repClus = self.Num2Clus[repClusInd]
            for idx in interactionList:
            # get the energy index here
                Interaction2En[idx] = Energies[self.clust2SpecClus[repClus][0]]
                # get the vector basis data here
                # if vector basis is empty, keep no of elements to -1.
                if self.clus2LenVecClus[self.clust2SpecClus[repClus][0]] == 0:
                    continue
                vecList = self.clust2vecClus[repClus]
                # store the number of vectors in the basis
                numVecsInteracts[idx] = len(vecList)
                # store the vector
                for vecidx, tup in enumerate(vecList):
                    VecsInteracts[idx, vecidx, :] = self.vecVec[tup[0]][tup[1]].copy()
                    VecGroupInteracts[idx, vecidx] = tup[0]
        print("Done with vector and energy data for interactions : {}".format(time.time() - start))

        return numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, numVecsInteracts, VecsInteracts,\
               VecGroupInteracts, numInteractsSiteSpec, SiteSpecInterArray

    def makeSiteIndToSite(self):
        Nsites = self.Nsites
        N_units = self.sup.superlatt[0, 0]
        siteIndtoR = np.zeros((Nsites, 3), dtype=int)
        RtoSiteInd = np.zeros((N_units, N_units, N_units), dtype=int)

        for siteInd in range(Nsites):
            R = self.sup.ciR(siteInd)[1]
            siteIndtoR[siteInd, :] = R
            RtoSiteInd[R[0], R[1], R[2]] = siteInd
        return siteIndtoR, RtoSiteInd