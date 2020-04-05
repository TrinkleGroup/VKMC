from onsager import crystal, cluster, supercell
import numpy as np
import collections
import itertools
import Transitions
import time


class ClusterSpecies():

    def __init__(self, specList, siteList):
        if len(specList)!= len(siteList):
            raise ValueError("Species and site lists must have same length")
        if not all(isinstance(site, cluster.ClusterSite) for site in siteList):
            raise TypeError("The sites must be entered as clusterSite object instances")
        # Form (site, species) set
        # Calculate the translation to bring center of the sites to the origin unit cell
        self.specList = specList
        self.siteList = siteList
        Rtrans = sum([site.R for site in siteList])//len(siteList)
        self.transPairs = [(site-Rtrans, spec) for site, spec in zip(siteList, specList)]
        self.SiteSpecs = set(self.transPairs)
        self.__hashcache__ = int(np.prod(np.array([hash((site, spec)) for site, spec in self.transPairs]))) +\
                             sum([hash((site, spec)) for site, spec in self.transPairs])

    def __eq__(self, other):
        if self.SiteSpecs == other.SiteSpecs:
            return True
        return False

    def __hash__(self):
        return self.__hashcache__

    def g(self, crys, g):
        return self.__class__(self.specList, [site.g(crys, g) for site in self.siteList])

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
    def __init__(self, sup, clusexp, jumpnetwork, mobCountList, vacSite):
        """
        :param sup : clusterSupercell object
        :param clusexp: cluster expansion about a single unit cell.
        :param mobList - list of labels for chemical species on mobile sites - in order as their occupancies are defined.
        :param sampleMobOccs - a starting mobile occupancy array to just count the number of each species
        :param vacSite - the site of the vacancy as a clusterSite object. This does not change during the simulation.
        In this type of simulations, we consider a solid with a single wyckoff set on which atoms are arranged.
        """
        self.chem = 0  # we'll work with a monoatomic basis
        self.sup = sup
        self.crys = self.sup.crys
        # vacInd will always be the initial state in the transitions that we consider.
        self.clusexp = clusexp
        self.mobCountList = mobCountList
        self.vacSpec = len(mobCountList) - 1
        self.mobList = list(range(len(mobCountList)))
        self.vacSite = vacSite  # This stays fixed throughout the simulation, so makes sense to store it.
        self.jumpnetwork = jumpnetwork
        self.ScalarBasis = self.createScalarBasis()
        self.SpecClusters = self.recalcClusters()
        self.vecClus, self.vecVec = self.genVecClustBasis(self.SpecClusters)
        self.indexVclus2Clus()

        # Generate the transitions-based data structures
        self.ijList, self.dxList, self.clustersOn, self.clustersOff = self.GetTransActiveClusts(self.jumpnetwork)

        # Generate the complete cluster basis including the arrangement of species on sites other than the vacancy site.
        self.KRAexpander = Transitions.KRAExpand(sup, self.chem, jumpnetwork, clusexp, mobCountList, vacSite)

    def recalcClusters(self):
        """
        Intended to take in a site based cluster expansion and recalculate the clusters with species in them
        """
        allClusts = set()
        symClusterList = []
        for clSet in self.clusexp:
            for clust in list(clSet):
                Nsites = len(clust.sites)
                occupancies = list(itertools.product(self.mobList, repeat=Nsites))
                for siteOcc in occupancies:
                    # Make the cluster site object
                    ClustSpec = ClusterSpecies(siteOcc, clust.sites)
                    # check if this has already been considered
                    if ClustSpec in allClusts:
                        continue
                    # Check if number of each species in the cluster is okay
                    mobcount = collections.Counter(siteOcc)
                    # Check if the number of atoms of a given species does not exceed the total number of atoms of that
                    # species in the solid.
                    if any(j > self.mobCountList[i] for i, j in mobcount.items()):
                        continue
                    # Otherwise, find all symmetry-grouped counterparts
                    newSymSet = set([ClustSpec.g(self.crys, g) for g in self.crys.G])
                    allClusts.update(newSymSet)
                    newList = list(newSymSet)
                    symClusterList.append(newList)

        return symClusterList

    def genVecClustBasis(self, specClusters):

        vecClustList = []
        vecVecList = []
        for clList in specClusters:
            cl0 = clList[0]
            glist0 = []
            for g in self.crys.G:
                if cl0.g(self.crys, g) == cl0:
                    glist0.append(g)

            G0 = sum([g.cartrot for g in glist0])/len(glist0)
            vals, vecs = np.linalg.eig(G0)
            vlist = [vecs[:, i]/np.linalg.norm(vecs[:, i]) for i in range(3) if np.isclose(vals[i], 1.0)]

            if len(vlist) == 0:  # For systems with inversion symmetry
                vecClustList.append(clList)
                vecVecList.append([np.zeros(3) for i in range(len(clList))])
                continue

            for v in vlist:
                newClustList = [cl0]
                # The first state being the same helps in indexing
                newVecList = [v]
                for g in self.crys.G:
                    cl1 = cl0.g(self.crys, g)
                    if cl1 in newClustList:
                        continue
                    newClustList.append(cl1)
                    newVecList.append(np.dot(g.cartrot, v))

                vecClustList.append(newClustList)
                vecVecList.append(newVecList)

        return vecClustList, vecVecList

    def createScalarBasis(self):
        """
        Function to add in the species arrangements to the cluster basis functions.
        """
        clusterBasis = []
        for clistInd, clist in enumerate(self.clusexp):
            cl0 = list(clist)[0]  # get the representative cluster
            # cluster.sites is a tuple, which maintains the order of the elements.
            Nmobile = len(cl0.sites)
            arrangemobs = itertools.product(self.mobList, repeat=Nmobile)  # arrange mobile sites on mobile species.
            for tup in arrangemobs:
                mobcount = collections.Counter(tup)
                # Check if the number of atoms of a given species does not exceed the total number of atoms of that
                # species in the solid.
                if any(j > self.mobCountList[i] for i, j in mobcount.items()):
                    continue
                # Each cluster is associated with three vectors
                clusterBasis.append((tup, clistInd))
        return clusterBasis

    def indexVclus2Clus(self):

        self.Vclus2Clus = np.zeros(len(self.vecClus), dtype=int)
        for vClusListInd, vClusList in enumerate(self.vecClus):
            clVec0 = vClusList[0]
            for cLlistInd, clList in enumerate(self.SpecClusters):
                cl0 = clList[0]
                if clVec0 == cl0:
                    self.Vclus2Clus[vClusListInd] = cLlistInd

    def Expand(self, beta, mobOccs, EnCoeffs, KRACoeffs):

        """
        :param beta : 1/KB*T
        :param mobOccs: the mobile occupancy in the current state
        :param EnCoeffs: energy interaction coefficients in a cluster expansion
        :param KRACoeffs: kinetic energy coefficients - pre-formed for each transition.
        :return: Wbar, Bbar - rate and bias expansions in the cluster basis
        """

        ijlist, dxlist = self.ijList, self.dxList

        del_lamb_mat = np.zeros((len(self.vecClus), len(self.vecClus), len(ijlist)))
        delxDotdelLamb = np.zeros((len(self.vecClus), len(ijlist)))

        # To be tensor dotted with ratelist with axes = (0,1)
        ratelist = np.zeros(len(ijlist))

        for (jnum, ij, dx) in zip(itertools.count(), ijlist, dxlist):

            del_lamb = np.zeros((len(self.vecClus), 3))

            specJ = sum([occ[ij[1]]*label for label, occ in enumerate(mobOccs)])
            indB, siteB = ij[1]
            indA, siteA = ij[0]

            if siteA != self.vacSite:
                raise ValueError("Incorrect initial jump site for vacancy. Got {}, expected {}".format(siteA,
                                                                                                       self.vacSite))
            # Get the KRA energy for this jump - stored in a dictionary element with key (indA, indB, specJ)
            delEKRA = self.KRAexpander.GetKRA((ij, dx), mobOccs, KRACoeffs[(indA, indB, specJ)])
            delE = 0.0  # This will added to the KRA energy to get the activation barrier

            # switch the occupancies in the final state
            mobOccs_final = mobOccs.copy()
            mobOccs_final[-1][ij[0]] = 0
            mobOccs_final[-1][ij[1]] = 1
            mobOccs_final[specJ][ij[0]] = 1
            mobOccs_final[specJ][ij[1]] = 0

            # (1) First, we deal with clusters that need to be switched off
            # (1.1) Check clusters that contain vacancy at vacSite.
            for clusterTupList in self.clustersOff[(self.vacSite, self.vacSpec)]:
                for clusterTup in clusterTupList:
                    # Check if the cluster is on
                    clust = clusterTup[2]
                    Rt = clusterTup[-1]
                    if all([mobOccs[spec][self.sup.index(site.R + Rt, site.ci)[0]] == 1
                            for site, spec in clust.SiteSpecs]):
                        del_lamb[clusterTup[0]] -= clusterTup[3]  # take away the vector associated with it.
                        delE -= EnCoeffs[self.Vclus2Clus[clusterTup[0]]]  # take away the energy coefficient

            # (1.2) Next, we deal with those clusters that have specJ at site B, and check if they are on.
            for clusterTupList in self.clustersOff[(siteB, specJ)]:
                for clusterTup in clusterTupList:
                    # Check if the cluster is on
                    clust = clusterTup[2]
                    Rt = clusterTup[-1]
                    if all([mobOccs[spec][self.sup.index(site.R + Rt, site.ci)[0]] == 1
                            for site, spec in clust.SiteSpecs]):
                        del_lamb[clusterTup[0]] -= clusterTup[3]  # take away the vector associated with it.
                        delE -= EnCoeffs[self.Vclus2Clus[clusterTup[0]]]  # take away the energy coefficient

            # (2) Now, we deal with clusters that need to be switched on
            # (2.1) - check clusters that contain the vacancy at siteB
            for clusterTupList in self.clustersOn[(siteB, self.vacSpec)]:
                for clusterTup in clusterTupList:
                    # Check if the cluster is on
                    clust = clusterTup[2]
                    Rt = clusterTup[-1]
                    if all([mobOccs_final[spec][self.sup.index(site.R + Rt, site.ci)[0]] == 1
                            for site, spec in clust.SiteSpecs]):
                        del_lamb[clusterTup[0]] += clusterTup[3]  # take away the vector associated with it.
                        delE += EnCoeffs[self.Vclus2Clus[clusterTup[0]]]  # add the energy coefficient

            # (2.2) - check clusters that contain the specJ at vacSite
            for clusterTupList in self.clustersOn[(self.vacSite, specJ)]:
                for clusterTup in clusterTupList:
                    # Check if the cluster is on
                    clust = clusterTup[2]
                    Rt = clusterTup[-1]
                    if all([mobOccs_final[spec][self.sup.index(site.R + Rt, site.ci)[0]] == 1
                            for site, spec in clust.SiteSpecs]):
                        del_lamb[clusterTup[0]] += clusterTup[3]  # take away the vector associated with it.
                        delE += EnCoeffs[self.Vclus2Clus[clusterTup[0]]]  # add the energy coefficient



            # for clListInd, clList in enumerate(self.vecClus):
            #     for clInd, clus in enumerate(clList):
            #         for siteSpec in clus.SiteSpecs:
            #             site, spec = siteSpec[0], siteSpec[1]
            #             # First, we check for vacancies
            #             # Check if any translation of this cluster needs to be turned off
            #             if site.ci == siteI and spec == len(mobOccs)-1:
            #                 siteSpecNew = set([(clsite - site.R, spec) for clsite, spec in clus.SiteSpecs])
            #                 # Check if this translated cluster is on in the initial state
            #                 if all([mobOccs[spec][self.sup.index(clSite.R, clSite.ci)[0]] == 1
            #                         for clSite, spec in siteSpecNew]):
            #                     # Turn in off
            #                     del_lamb[clListInd] -= self.vecVec[clListInd][clInd]
            #                     delE -= EnCoeffs[self.Vclus2Clus[clListInd]]
            #
            #                 # Check if this cluster is on in the final state
            #                 if all([mobOccs_final[spec][self.sup.index(clSite.R, clSite.ci)[0]] == 1
            #                         for clSite, spec in siteSpecNew]):
            #                     # Turn it on
            #                     del_lamb[clListInd] += self.vecVec[clListInd][clInd]
            #                     delE += EnCoeffs[self.Vclus2Clus[clListInd]]
            #
            #             # Next, we check for specJ
            #             if site.ci == siteJ.ci and spec == specJ:
            #                 # Bring this site to Rj instead of site.R
            #                 siteSpecNew = set([(clsite - site.R + siteJ.R, spec) for clsite, spec in clus.SiteSpecs])
            #                 # Check if this translated cluster is on in the initial state
            #                 if all([mobOccs[spec][self.sup.index(clSite.R, clSite.ci)[0]] == 1
            #                         for clSite, spec in siteSpecNew]):
            #                     # Turn it off
            #                     del_lamb[clListInd] -= self.vecVec[clListInd][clInd]
            #                     delE -= EnCoeffs[self.Vclus2Clus[clListInd]]
            #
            #                 # Check if this cluster is on in the final state
            #                 if all([mobOccs_final[spec][self.sup.index(clSite.R, clSite.ci)[0]] == 1
            #                         for clSite, spec in siteSpecNew]):
            #                     # Turn it on
            #                     del_lamb[clListInd] -= self.vecVec[clListInd][clInd]
            #                     delE -= EnCoeffs[self.Vclus2Clus[clListInd]]
            #
            # append to the rateList
            ratelist[jnum] = np.exp(-(0.5*delE + delEKRA))

            # Create the matrix to find Wbar
            del_lamb_mat[:, :, jnum] = np.dot(del_lamb, del_lamb.T)

            # Create the matrix to find Bbar
            delxDotdelLamb[:, jnum] = np.tensordot(del_lamb_mat, dx, axes=(1, 0))

        Wbar = np.tensordot(ratelist, del_lamb_mat, axes=(0, 0))
        Bbar = np.tensordot(ratelist, delxDotdelLamb, axes=(0, 1))

        return Wbar, Bbar

    def GetTransActiveClusts(self, jumpnetwork):
        """
        Function to find and store those clusters that wither turn on or off due to transitons
        out of a given state
        :param jumpnetwork: vacancy jumpnetwork
        """
        ijList = []
        dxList = []
        clusterTransOff = collections.defaultdict(list)  # these are clusters that need to be turned off
        clusterTransOn = collections.defaultdict(list)  # these are clusters that need to be turned off

        # pre-process those clusters that contain vacancy at vacSite.
        for vclusListInd, clustList, vecList in zip(itertools.count(), self.vecClus, self.vecVec):
            for clInd, clust, vec in zip(itertools.count(), clustList, vecList):
                for site, spec in clust.SiteSpecs:
                    # see if the cluster has a vacancy site with the vacancy on it
                    if site.ci == self.vacSite.ci:
                        if spec == self.vacSpec:
                            clusterTransOff[(self.vacSite, self.vacSpec)].append((vclusListInd, clInd, clust, vec,
                                                                                  self.vacSite.R - site.R))
                        elif spec != self.vacSpec:
                            # There is some other species at the vacancy site, must be the final state of some jump.
                            clusterTransOn[(self.vacSite, spec)].append((vclusListInd, clInd, clust, vec,
                                                                                  self.vacSite.R - site.R))

        for jump in [jmp for jList in jumpnetwork for jmp in jList]:
            siteA = cluster.ClusterSite(ci=(self.chem, jump[0][0]), R=np.zeros(3, dtype=int))
            if siteA != self.vacSite:
                # if the initial site of the vacancy is not the vacSite, it is not a jump out of this state.
                # Ignore it - removes reverse jumps from multi-site, single-Wyckoff lattices.
                continue
            Rj, (c, cj) = self.crys.cart2pos(jump[1] -
                                             np.dot(self.crys.lattice, self.crys.basis[self.chem][jump[0][1]]) +
                                             np.dot(self.crys.lattice, self.crys.basis[self.chem][jump[0][0]]))
            # check we have the correct site
            if not cj == jump[0][1]:
                raise ValueError("improper coordinate transformation, did not get same final jump site")
            siteB = cluster.ClusterSite(ci=(self.chem, jump[0][1]), R=Rj)

            indA = self.sup.index(siteA.R, siteA.ci)[0]
            indB = self.sup.index(siteB.R, siteB.ci)[0]
            ijList.append(((indA, siteA), (indB, siteB)))
            dxList.append(jump[1])
            # See which clusters contain specJ at siteJ
            for vclusListInd, clustList, vecList in zip(itertools.count(), self.vecClus, self.vecVec):
                for clInd, clust, vec in zip(itertools.count(), clustList, vecList):
                    for site, spec in clust.SiteSpecs:
                        # see if the cluster has a vacancy site with the vacancy on it
                        if site.ci == siteB.ci:
                            if spec != self.vacSpec:
                                # translate all the sites and see if the vacancy is in there as well
                                # if yes, We will not consider this to prevent double counting
                                Rt = siteB.R - site.R  # to make the sites coincide
                                if not (self.vacSite, self.vacSpec) in [(site + Rt, spec)
                                                                        for (site, spec) in clust.SiteSpecs]:
                                    clusterTransOff[(siteB, spec)].append((vclusListInd, clInd, clust, vec, Rt))
                            else:
                                # if the cluster contains a vacancy at siteB.ci
                                Rt = siteB.R - site.R
                                # Check for double counting
                                if not self.vacSite in [site + Rt for site, spec in clust.SiteSpecs]:
                                    # if vacSite IS present, then this means that there is some other species
                                    # on it, which has already been accounted for previously.
                                    clusterTransOn[(siteB, self.vacSpec)].append((vclusListInd, clInd, clust, vec,
                                                                                  Rt))

        return ijList, dxList, clusterTransOn, clusterTransOff













