from onsager import crystal, supercell, cluster
import numpy as np
import collections
import itertools
from tqdm import tqdm
import ClustSpec
from ClustSpec import ClusterSpecies
import Cluster_Expansion
import unittest

class test_ClusterSpecies(unittest.TestCase):
    def test_equality(self):
        spec1 = 0
        spec2 = 1
        site1 = cluster.ClusterSite(ci=(0, 0), R=np.zeros(3, dtype=int))
        site2 = cluster.ClusterSite(ci=(0, 0), R=np.ones(3, dtype=int))
        siteList1 = [site1, site2]
        siteList2 = [site2, site1]
        specListDiff = [spec1, spec2]
        specListSame1 = [spec1, spec1]
        specListSame2 = [spec2, spec2]

        cl1 = ClusterSpecies(specListDiff, siteList1)
        cl2 = ClusterSpecies(specListDiff, siteList2)
        self.assertNotEqual(cl1, cl2)

        # same species but swapped sites should give the same cluster
        cl1 = ClusterSpecies(specListSame1, siteList1)
        cl2 = ClusterSpecies(specListSame1, siteList2)
        self.assertEqual(cl1, cl2)

        cl1 = ClusterSpecies(specListSame2, siteList1)
        cl2 = ClusterSpecies(specListSame2, siteList2)
        self.assertEqual(cl1, cl2)
        print("Equality checks okay")


class test_Vector_Cluster_Expansion(unittest.TestCase):

    def test_recalcClusters(self):
        # check that every cluster appears just once in just one symmetry list
        clList = [cl for clList in self.VclusExp.SpecClusters for cl in clList]
        self.assertEqual(len(clList), len(set(clList)))
        TwoBodyCount = 0
        ThreeBodyCount = 0
        OneBodyCount = 0
        for cl in clList:
            if len(cl.SiteSpecs) == 1:
                OneBodyCount += 1

            if len(cl.SiteSpecs) == 2:
                TwoBodyCount += 1

            if len(cl.SiteSpecs) == 3:
                ThreeBodyCount += 1

        if self.crtype == "FCC":
            print("Checking FCC")
            self.assertEqual(OneBodyCount, self.NSpec)
            two = (6 + 3) * (2 * (self.NSpec - 1) + (self.NSpec - 1) ** 2)
            # 6 translationally unique 1nn pairs, 3 for 2nn pairs,
            # then first term for vacancy-spec clusters, and second term for spec-spec clusters
            self.assertEqual(two, TwoBodyCount)

            three = (12 + 8) * (3 * (self.NSpec - 1) ** 2 + (self.NSpec - 1) ** 3)
            # 12 translationally unique triplets on the cube faces, with max allowed separation of a0 between any two,
            # and 8 on the octahedral cage faces.
            self.assertEqual(three, ThreeBodyCount)

        elif self.crtype=="FCC_orthogonal":
            print("Checking cubic FCC")

            # For the cubic FCC lattice, a CE will be built around each of the basis sites, so we'll have 4 times
            # as many as the primitive case

            self.assertEqual(OneBodyCount, 4 * self.NSpec)

            two = (6 + 3) * (2 * (self.NSpec - 1) + (self.NSpec - 1) ** 2)
            self.assertEqual(4 * two, TwoBodyCount)

            three = (12 + 8) * (3 * (self.NSpec - 1) ** 2 + (self.NSpec - 1) ** 3)
            self.assertEqual(4 * three, ThreeBodyCount)

        elif self.crtype == "BCC":
            print("Checking BCC")
            self.assertEqual(OneBodyCount, self.NSpec)
            two = (4 + 3) * (2 * (self.NSpec - 1) + (self.NSpec - 1) ** 2)
            self.assertEqual(two, TwoBodyCount)

            three = 12 * (3 * (self.NSpec - 1) ** 2 + (self.NSpec - 1) ** 3)  # 12 pyramidal triplets in BCC
            self.assertEqual(three, ThreeBodyCount)

        print(OneBodyCount, TwoBodyCount, ThreeBodyCount)

        # let's test if the number of symmetric site clusters generated is the same
        siteSets = []
        for clusterList in self.VclusExp.SpecClusters:
            # First check symmetry grouping
            for cl0 in clusterList:
                for g in self.VclusExp.crys.G:
                    clNew = cl0.g(self.crys, g)
                    self.assertTrue(clNew in clusterList)
                    # check that symmetrically rotated clusters have the same species
                    site0List = cl0.siteList
                    sitesRot = [site.g(self.VclusExp.crys, g) for site in site0List]
                    Rtrans = sum(site.R for site in sitesRot) // len(sitesRot)
                    if self.VclusExp.zeroClusts:
                        sitesRot = [site - Rtrans for site in sitesRot]
                    self.assertEqual(sitesRot, clNew.siteList)
                    self.assertEqual(cl0.specList, clNew.specList)

            for clust in clusterList:
                Rtrans = sum([site.R for site in clust.siteList]) // len(clust.siteList)
                sites_shift = tuple([site - Rtrans for site in clust.siteList])
                # check correct translations
                sitesBuilt = tuple([site for site, spec in clust.transPairs])
                if self.VclusExp.zeroClusts:
                    self.assertEqual(sites_shift, sitesBuilt)
                st = set(sites_shift)
                if st not in siteSets:
                    siteSets.append(st)

        sitesFromClusexp = []
        for clSet in self.VclusExp.clusexp:
            for cl in list(clSet):
                Rtrans = sum([site.R for site in cl.sites]) // len(cl.sites)
                sitesFromClusexp.append(set(tuple([site - Rtrans for site in cl.sites])))

        total_reps = len(sitesFromClusexp)
        self.assertEqual(len(sitesFromClusexp), len(siteSets), msg="{}, {}".format(total_reps, len(siteSets)))

        for ClSiteSet in siteSets:
            self.assertTrue(ClSiteSet in sitesFromClusexp)

        # Go through all the site clusters:
        total_spec_clusts = 0
        tot_1 = 0
        tot_2 = 0
        tot_3 = 0

        for clSites in sitesFromClusexp:
            ordr = len(clSites)
            # for singletons, any species assignment is acceptable
            if ordr == 1:
                total_spec_clusts += self.NSpec
                tot_1 += self.NSpec
                continue

            # Now, for two and three body clusters:
            # we can have max one vacancy at a site, and any other site can have any other occupancy.
            # or we can have no vacancy at all in which case it's just (Nspec - 1)**ordr
            total_spec_clusts += (ordr * (self.NSpec - 1) ** (ordr - 1) + (self.NSpec - 1) ** ordr)
            if ordr == 2:
                tot_2 += (ordr * (self.NSpec - 1) ** (ordr - 1) + (self.NSpec - 1) ** ordr)

            if ordr == 3:
                tot_3 += (ordr * (self.NSpec - 1) ** (ordr - 1) + (self.NSpec - 1) ** ordr)

        self.assertEqual(tot_1, OneBodyCount)
        self.assertEqual(tot_2, TwoBodyCount)
        self.assertEqual(tot_3, ThreeBodyCount)
        total_code = sum([len(lst) for lst in self.VclusExp.SpecClusters])
        self.assertEqual(total_code, total_spec_clusts, msg="{}, {}".format(total_code, total_spec_clusts))
        print("Done species assignment tests")

    def test_genVecClustBasis(self):
        # Here, we test if we have generated the vector cluster basis (site-based only) properly

        for clListInd, clList, vecList in zip(itertools.count(), self.VclusExp.vecClus,
                                              self.VclusExp.vecVec):
            self.assertEqual(len(clList), len(vecList))
            cl0, vec0 = clList[0], vecList[0]
            for clust, vec in zip(clList, vecList):
                # Check that symmetry operations are consistent
                self.assertFalse(np.iscomplexobj(vec))  # check that all vectors are real.
                count = 0
                for g in self.crys.G:
                    if cl0.g(self.crys, g) == clust:
                        count += 1
                        self.assertTrue(np.allclose(np.dot(g.cartrot, vec0), vec),
                                        msg="\n{}, {} \n{}, {}\n{}\n{}".format(vec0, vec, cl0, clust,
                                                                               self.crys.lattice, g.cartrot))
                self.assertGreater(count, 0)

                # Now check invariance of basis vectors
                count = 0
                for g in self.crys.G:
                    if cl0.g(self.crys, g) == cl0:
                        count += 1
                        self.assertTrue(np.allclose(np.dot(g.cartrot, vec0), vec0),
                                        msg="\n{}, {} \n{}\n{}".format(vec0, cl0, self.crys.lattice, g.cartrot))

                # There must be at least one group op that keeps the cluster invariant to be considered in vector
                # groups
                self.assertGreater(count, 0)

    def testcluster2vecClus(self):

        # check that every cluster appears just once in just one symmetry list
        clList = [cl for clList in self.VclusExp.SpecClusters for cl in clList]

        for cl1 in clList:
            count = 0
            for cl2 in clList:
                if cl1 == cl2:
                    count += 1
            self.assertEqual(count, 1)  # check that the cluster occurs only once

        for clListInd, clList in enumerate(self.VclusExp.SpecClusters):
            for clust in clList:
                if self.VclusExp.clus2LenVecClus[clListInd] == 0:
                    self.assertFalse(clust in self.VclusExp.clust2vecClus.keys())
                else:
                    self.assertTrue(self.VclusExp.clus2LenVecClus[clListInd] <= 3)
                    vecList = self.VclusExp.clust2vecClus[clust]
                    self.assertTrue(self.VclusExp.clus2LenVecClus[clListInd] == len(vecList))
                    for tup in vecList:
                        self.assertEqual(clust, self.VclusExp.vecClus[tup[0]][tup[1]])
                        vec0 = self.VclusExp.vecVec[tup[0]][tup[1]]
                        # check that that vector is invariant under group operations that leave the cluster unchanges
                        count = 0
                        for g in self.crys.G:
                            if clust.g(self.crys, g) == clust:
                                count += 1
                                self.assertTrue(np.allclose(np.dot(g.cartrot, vec0), vec0),
                                                msg="\n{}, {} \n{}\n{}".format(vec0, clust, self.crys.lattice,
                                                                               g.cartrot))
                        # check that at least one group op was found that satisfied the test
                        self.assertGreater(count, 0)

    def test_indexClusterGroupToVectorGroup(self):
        for vclusListInd, clListInd in enumerate(self.VclusExp.Vclus2Clus):
            cl0 = self.VclusExp.vecClus[vclusListInd][0]
            self.assertEqual(cl0, self.VclusExp.SpecClusters[clListInd][0])

    def test_generateSiteSpecInteracts(self):
        # test that every interaction is valid
        # The key site should be present only once in every interaction stored for it
        interactCounter = collections.defaultdict(int)
        if self.reqSites is None:
            self.assertEqual(len(self.VclusExp.SiteSpecInteractIds), self.NSpec * len(self.superCell.mobilepos))
        else:
            print("checking required site presence.")

        for (key, interactIdList) in self.VclusExp.SiteSpecInteractIds.items():
            siteInd = key[0]
            sp = key[1]
            for Id in interactIdList:
                interaction = self.VclusExp.Id2InteractionDict[Id]
                if self.reqSites is not None:
                    req_count = 0
                    for (site, spec) in interaction:
                        if site in self.reqSites:
                            req_count += 1

                    self.assertTrue(req_count >= 1)

                interactCounter[interaction] += 1
                count = 0
                for (site, spec) in interaction:
                    if site == siteInd and sp == spec:
                        count += 1
                self.assertEqual(count, 1)

        interactCounter.default_factory = None
        # Check that an interaction has occurred as many times as there are sites in it
        for interact, count in interactCounter.items():
            self.assertEqual(count, len(interact))

        # check that all representative clusters have been translated
        self.assertEqual(len(self.VclusExp.clust2InteractId),
                         sum([len(clList) for clList in self.VclusExp.SpecClusters]),
                         msg="{}".format(len(self.VclusExp.clust2InteractId)))

        clAll = [cl for clList in self.VclusExp.SpecClusters for cl in clList]
        for clInd, cl in enumerate(clAll):
            self.assertEqual(clInd, self.VclusExp.Clus2Num[cl])

        for clustInd in self.VclusExp.clust2InteractId.keys():
            clust = self.VclusExp.Num2Clus[clustInd]
            self.assertTrue(clust in clAll)

        # check that each interaction corresponds to a single rep cluster
        InteractIdtoRepClust = collections.defaultdict(list)
        for repClust, interactIdList in self.VclusExp.clust2InteractId.items():
            for interId in interactIdList:
                InteractIdtoRepClust[interId].append(repClust)

        for key, item in InteractIdtoRepClust.items():
            self.assertEqual(len(item), 1)

        # test that all translations of all representative clusters are considered
        allSpCl = [SpCl for SpClList in self.VclusExp.SpecClusters for SpCl in SpClList]
        for (key, interactIdList) in self.VclusExp.SiteSpecInteractIds.items():
            siteInd = key[0]
            ci, Rsite = self.VclusExp.sup.ciR(siteInd)
            sp = key[1]
            if self.reqSites is None:
                count = 0
                # For each species at a site, check that all translations of a cluster are considered.
                for SpecClus in allSpCl:
                    for (site, spec) in SpecClus.SiteSpecs:
                        if spec == sp and site.ci == ci:
                            count += 1  # a translation of this cluster should exist

                self.assertEqual(count, len(interactIdList),
                                 msg="\ncount {}, stored {}\n{}".format(count, len(interactIdList), key))

            else:
                count = 0
                # For each species at a site, check that all translations of a cluster are considered.
                for SpecClus in allSpCl:
                    for (site, spec) in SpecClus.SiteSpecs:
                        if spec == sp and site.ci == ci:
                            # get the translation
                            Rshift = Rsite - site.R
                            # translate all the sites by this translation
                            reqSitesFound = 0
                            for (clSite, clSpec) in SpecClus.SiteSpecs:
                                RClSiteNew = clSite.R + Rshift
                                clSiteIndNew, _ = self.superCell.index(RClSiteNew, clSite.ci)
                                if clSiteIndNew in self.reqSites:
                                    reqSitesFound += 1

                            # the translated cluster must have a required site to be counted
                            if reqSitesFound > 0:
                                count += 1

                self.assertEqual(count, len(interactIdList),
                                 msg="\ncount {}, stored {}\n{}".format(count, len(interactIdList), key))

    def testcluster2SpecClusGroup(self):

        for clListInd, clList in enumerate(self.VclusExp.SpecClusters):
            for clustInd, clust in enumerate(clList):
                tup = self.VclusExp.clust2SpecClus[clust]
                self.assertEqual(tup[0], clListInd)
                self.assertEqual(tup[1], clustInd)

    def test_arrays(self):
        # First, we generate the Jit arrays

        # First, the chemical data
        numSitesInteracts, SupSitesInteracts, SpecOnInteractSites, Interaction2En, \
        numInteractsSiteSpec, SiteSpecInterArray = \
            self.VclusExp.makeJitInteractionsData(self.Energies)

        # Next, the vector basis data
        numVecsInteracts, VecsInteracts, VecGroupInteracts = self.VclusExp.makeJitVectorBasisData()

        # check that every representative clusters has been translated as many times
        # as there are unit cells in the super cell.

        allSpCl = [SpCl for SpClList in self.VclusExp.SpecClusters for SpCl in SpClList]

        if self.reqSites is None:
            # In all our use cases, chemistry 0 is the only chemistry
            allCount = len(self.VclusExp.sup.mobilepos) * len(allSpCl) // len(self.crys.basis[self.VclusExp.chem])
            self.assertEqual(len(self.VclusExp.Id2InteractionDict), allCount,
                             msg="\n{}\n{}".format(len(self.VclusExp.Id2InteractionDict), allCount))
            self.assertEqual(allCount, len(numSitesInteracts))

        allSpClset = set(allSpCl)
        self.assertEqual(len(allSpCl), len(allSpClset))
        self.assertEqual(len(SupSitesInteracts), len(numSitesInteracts))
        self.assertEqual(len(SpecOnInteractSites), len(numSitesInteracts))

        repclustCount = collections.defaultdict(int)

        SpClset = set([])

        print("checking interactions", flush=True)
        for i in tqdm(range(len(numSitesInteracts)), position=0, leave=True):
            siteSpecSet = set([(SupSitesInteracts[i, j], SpecOnInteractSites[i, j])
                               for j in range(numSitesInteracts[i])])
            interaction = self.VclusExp.Id2InteractionDict[i]
            interactionSet = set(interaction)
            self.assertEqual(siteSpecSet, interactionSet)

            # let's get the representative cluster of this interaction
            repClusStored = self.VclusExp.Num2Clus[self.VclusExp.InteractionId2ClusId[i]]
            siteList = []
            specList = []

            for siteInd, spec in [(SupSitesInteracts[i, j], SpecOnInteractSites[i, j])
                                  for j in range(numSitesInteracts[i])]:
                ci, R = self.VclusExp.sup.ciR(siteInd)
                clSite = cluster.ClusterSite(ci=ci, R=R)
                siteList.append(clSite)
                specList.append(spec)

            SpCl = ClusterSpecies(specList, siteList)

            # apply periodicity to the siteList and rebuild
            siteListBuilt = [site for (site, spec) in SpCl.transPairs]
            specListBuilt = [spec for (site, spec) in SpCl.transPairs]
            siteListnew = []
            for site in siteListBuilt:
                R = site.R
                ci = site.ci
                idxSite, _ = self.VclusExp.sup.index(R, ci)
                ciNew, RNew = self.VclusExp.sup.ciR(idxSite)
                self.assertEqual(ciNew, ci)
                siteListnew.append(cluster.ClusterSite(ci=ci, R=RNew))

            SpCl = ClusterSpecies(specListBuilt, siteListnew)
            self.assertEqual(SpCl, repClusStored)
            self.assertTrue(SpCl in allSpClset, msg="\n{}\n{}\n{}\n{}".format(SpCl, repClusStored, siteList, specList))

            # Check that the correct energies have been assigned
            En = self.Energies[self.VclusExp.clust2SpecClus[repClusStored][0]]
            EnStored = Interaction2En[i]
            self.assertAlmostEqual(En, EnStored, 10)
            SpClset.add(SpCl)

            repclustCount[SpCl] += 1

        # Check that all repclusts were considered
        self.assertEqual(SpClset, allSpClset)
        # Check that all translations of the clusters were considered
        for key, item in repclustCount.items():
            if self.reqSites is None:
                self.assertEqual(item, len(self.VclusExp.sup.mobilepos) // len(self.crys.basis[self.VclusExp.chem]))

        print("checked interactions \n checking vectors", flush=True)

        # Now, test the vector basis and energy information for the clusters
        for i in tqdm(range(len(numSitesInteracts)), position=0, leave=True):
            # get the representative cluster
            repClus = self.VclusExp.Num2Clus[self.VclusExp.InteractionId2ClusId[i]]

            # test the energy index
            enIndex = self.VclusExp.clust2SpecClus[repClus][0]
            self.assertEqual(Interaction2En[i], self.Energies[enIndex])

            # if the vector basis is empty, continue
            if self.VclusExp.clus2LenVecClus[self.VclusExp.clust2SpecClus[repClus][0]] == 0:
                self.assertEqual(numVecsInteracts[i], -1)
                continue
            # get the vector basis info for this cluster
            vecList = self.VclusExp.clust2vecClus[repClus]
            # check the number of vectors
            self.assertEqual(numVecsInteracts[i], len(vecList))
            # check that the correct vector have been stored, in the same order as in vecList (not necessary but short testing)
            for vecind in range(len(vecList)):
                vec = self.VclusExp.vecVec[vecList[vecind][0]][vecList[vecind][1]]
                self.assertTrue(np.allclose(vec, VecsInteracts[i, vecind, :]))
                count = 0
                for g in self.crys.G:
                    if repClus.g(self.crys, g) == repClus:
                        count += 1
                        self.assertTrue(np.allclose(np.dot(g.cartrot, vec), vec),
                                        msg="\n{}, {} \n{}\n{}".format(vec, repClus, self.crys.lattice, g.cartrot))
                # check that at least one group op was found that satisfied the test
                self.assertGreater(count, 0)

        print("checked vectors.", flush=True)

        # Next, test the interactions each (site, spec) is a part of
        self.assertEqual(numInteractsSiteSpec.shape[0], len(self.VclusExp.sup.mobilepos))
        self.assertEqual(numInteractsSiteSpec.shape[1], self.NSpec)
        for siteInd in range(len(self.superCell.mobilepos)):
            for spec in range(self.NSpec):
                numInteractStored = numInteractsSiteSpec[siteInd, spec]
                # get the actual count
                # ci, R = self.VclusExp.sup.ciR(siteInd)
                # if len(self.reqSites) == 0:
                if (siteInd, spec) in self.VclusExp.SiteSpecInteractIds.keys():
                    self.assertEqual(len(self.VclusExp.SiteSpecInteractIds[(siteInd, spec)]), numInteractStored,
                                     msg="key: {}, len: {}, stored_array: {}".format((siteInd, spec),
                                                                                     len(
                                                                                         self.VclusExp.SiteSpecInteractIds[
                                                                                             (siteInd, spec)]),
                                                                                     numInteractStored))

                else:
                    self.assertEqual(0, numInteractStored, msg="key: {}, stored_array: {}".format((siteInd, spec),
                                                                                                  numInteractStored))

                for IdxOfInteract in range(numInteractStored):
                    interactMainIndex = SiteSpecInterArray[siteInd, spec, IdxOfInteract]
                    self.assertEqual(interactMainIndex,
                                     self.VclusExp.SiteSpecInteractIds[(siteInd, spec)][IdxOfInteract])


class test_BCC(test_Vector_Cluster_Expansion):

    def setUp(self):
        self.NSpec = 3
        self.Nvac = 1
        self.vacSpec = 2
        self.MaxOrder = 3
        a0 = 1
        self.a0 = a0
        self.crtype = "FCC"
        self.crys = crystal.Crystal.BCC(a0, chemistry="A")
        self.N_units = 8
        self.superlatt = self.N_units * np.eye(3, dtype=int)
        self.superCell = supercell.ClusterSupercell(self.crys, self.superlatt)
        # get the number of sites in the supercell - should be 8x8x8
        numSites = len(self.superCell.mobilepos)
        self.vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
        self.vacsiteInd = self.superCell.index(np.zeros(3, dtype=int), (0, 0))[0]
        self.clusexp = cluster.makeclusters(self.crys, 1.01*a0, self.MaxOrder)

        # sup, clusexp, NSpec, vacSite, vacSpec, maxorder, Nvac=1, MadeSpecClusts=None, zeroClusts=True
        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(self.superCell, self.clusexp, self.NSpec,
                                                                 self.vacsite, self.vacSpec, self.MaxOrder)

        self.reqSites = None #[i for i in range(10)]  # All interactions must have at least one of these sites
        self.VclusExp.generateSiteSpecInteracts(reqSites=self.reqSites)

        self.VclusExp.genVecClustBasis(self.VclusExp.SpecClusters)
        self.VclusExp.indexVclus2Clus()  # Index vector cluster list to cluster symmetry groups
        self.VclusExp.indexClustertoVecClus()

        self.Energies = np.random.rand(len(self.VclusExp.SpecClusters))
        print("Done setting up BCC cluster expansion tests.")

    def testBCC_counts(self):
        # This test is specifically for a BCC, 3-spec, 3-order, 2nn cluster expansion
        oneCounts = 0
        twocounts = 0
        threecounts = 0

        oneBody = 0
        TwoBody = 0
        ThreeBody = 0

        oneBodyList = []
        TwoBodyList = []
        ThreeBodyList = []

        sitesFromClusexp = set([])
        for clSet in self.clusexp:
            for cl in list(clSet):
                for g in self.VclusExp.crys.G:
                    newsitelist = tuple([site.g(self.VclusExp.crys, g) for site in cl.sites])
                    Rtrans = sum([site.R for site in newsitelist]) // len(newsitelist)
                    sitesFromClusexp.add(tuple([site - Rtrans for site in newsitelist]))

        for siteList in sitesFromClusexp:
            ln = len(siteList)
            if ln == 1:
                oneCounts += 1
            elif ln == 2:
                twocounts += 1
            elif ln == 3:
                threecounts += 1

        self.assertEqual((oneCounts, twocounts, threecounts), (1, 14, 24))
        print("Checking cluster counts")
        for clusterList in self.VclusExp.SpecClusters:
            for clust in clusterList:
                order = len(clust.SiteSpecs)
                if order == 1:
                    oneBody += 1
                    oneBodyList.append(clust)
                if order == 2:
                    TwoBody += 1
                    TwoBodyList.append(clust)

                if order == 3:
                    ThreeBody += 1
                    ThreeBodyList.append(clust)

        self.assertEqual(oneBody, 3)
        self.assertEqual(TwoBody, 56)
        self.assertEqual(ThreeBody, 240)

        # Now let's check some two body numbers
        vacCount = 0
        nonVacCount = 0
        for clust in TwoBodyList:
            specList = clust.specList
            if any([spec == 2 for spec in specList]):
                vacCount += 1
            else:
                nonVacCount += 1

        self.assertEqual(vacCount,
                         2 * 2 * 4 + 2 * 2 * 3)  # The first term is for nearest neighbor, the second for second nearest neighbor
        self.assertEqual(nonVacCount, 2 * 2 * 4 + 2 * 2 * 3)

        # Now let's check some three body numbers
        vacCount = 0
        nonVacCount = 0
        for clust in ThreeBodyList:
            specList = clust.specList
            if any([spec == 2 for spec in specList]):
                vacCount += 1
            else:
                nonVacCount += 1

        self.assertEqual(vacCount, (3 * 2 * 2) * 12)
        self.assertEqual(nonVacCount, (2 * 2 * 2) * 12)


class test_FCC(test_Vector_Cluster_Expansion):

    def setUp(self):
        self.NSpec = 3
        self.Nvac = 1
        self.vacSpec = 2
        self.MaxOrder = 3
        a0 = 1
        self.a0 = a0
        self.crtype="FCC"
        self.crys = crystal.Crystal.FCC(a0, chemistry="A")

        self.N_units = 8
        self.superlatt = self.N_units * np.eye(3, dtype=int)
        self.superCell = supercell.ClusterSupercell(self.crys, self.superlatt)
        # get the number of sites in the supercell - should be 8x8x8
        numSites = len(self.superCell.mobilepos)
        self.vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
        self.vacsiteInd = self.superCell.index(np.zeros(3, dtype=int), (0, 0))[0]
        self.clusexp = cluster.makeclusters(self.crys, 1.01*a0, self.MaxOrder)
        numClusts = sum(len(clSet) for clSet in self.clusexp)
        print("Number of site clusters: {}".format(numClusts))

        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(self.superCell, self.clusexp, self.NSpec,
                                                                 self.vacsite, self.vacSpec, self.MaxOrder)

        self.reqSites = None  #[i for i in range(10)]  # All interactions must have at least one of these sites
        self.VclusExp.generateSiteSpecInteracts(reqSites=self.reqSites)

        self.VclusExp.genVecClustBasis(self.VclusExp.SpecClusters)
        self.VclusExp.indexVclus2Clus()  # Index vector cluster list to cluster symmetry groups
        self.VclusExp.indexClustertoVecClus()

        self.Energies = np.random.rand(len(self.VclusExp.SpecClusters))
        print("Done setting up FCC cluster expansion tests.")

    def test_FCC_BasisVectors(self):
        dx = np.array([0.5, 0., 0.5]) * self.a0
        R, ci = self.crys.cart2pos(dx)
        site1 = cluster.ClusterSite(ci=ci, R=R)
        R2, ci2 = self.crys.cart2pos(np.zeros(3))
        site2 = cluster.ClusterSite(ci=ci2, R=R2)

        # choose any species
        spec1 = self.NSpec - 1
        spec2 = self.NSpec - 2
        specList = (spec1, spec2)
        siteList = (site1, site2)

        spCl = ClustSpec.ClusterSpecies(specList, siteList)

        clList = [cl for clList in self.VclusExp.SpecClusters for cl in clList]
        self.assertTrue(spCl in clList)
        clSymGroup = self.VclusExp.clust2SpecClus[spCl][0]
        self.assertEqual(len(self.VclusExp.SpecClusters[clSymGroup]), 12)
        NumVecGroups = self.VclusExp.clus2LenVecClus[clSymGroup]
        self.assertEqual(NumVecGroups, 1)
        vectorGroup, position = self.VclusExp.clust2vecClus[spCl][0]
        print(dx, self.VclusExp.vecVec[vectorGroup][position])
        print()

        # Let's check out a straight axis
        dx = np.array([1., 0., 0.]) * self.a0
        R, ci = self.crys.cart2pos(dx)
        site1 = cluster.ClusterSite(ci=ci, R=R)
        R2, ci2 = self.crys.cart2pos(np.zeros(3))
        site2 = cluster.ClusterSite(ci=ci2, R=R2)

        specList = (spec1, spec2)
        siteList = (site1, site2)

        spCl = ClustSpec.ClusterSpecies(specList, siteList)

        clList = [cl for clList in self.VclusExp.SpecClusters for cl in clList]
        self.assertTrue(spCl in clList)
        clSymGroup = self.VclusExp.clust2SpecClus[spCl][0]
        self.assertEqual(len(self.VclusExp.SpecClusters[clSymGroup]), 6)
        NumVecGroups = self.VclusExp.clus2LenVecClus[clSymGroup]
        self.assertEqual(NumVecGroups, 1)
        vectorGroup, position = self.VclusExp.clust2vecClus[spCl][0]
        self.assertEqual(len(self.VclusExp.vecVec[vectorGroup]), 6)
        print(dx, self.VclusExp.vecVec[vectorGroup][position])
        print()
        # Let's check out a 3-body cluster on the face - mirror symmetry
        dx1 = np.array([1., 0., 0.]) * self.a0
        dx2 = np.array([0.5, 0.5, 0.]) * self.a0
        R1, ci1 = self.crys.cart2pos(dx1)
        R2, ci2 = self.crys.cart2pos(dx2)
        R3, ci3 = self.crys.cart2pos(np.zeros(3))
        site1 = cluster.ClusterSite(ci=ci1, R=R1)
        site2 = cluster.ClusterSite(ci=ci2, R=R2)
        site3 = cluster.ClusterSite(ci=ci3, R=R3)
        spec3 = self.NSpec - 3
        specList = (spec1, spec2, spec3)
        siteList = (site1, site2, site3)

        spCl = ClustSpec.ClusterSpecies(specList, siteList)

        clList = [cl for clList in self.VclusExp.SpecClusters for cl in clList]
        self.assertTrue(spCl in clList)
        clSymGroup = self.VclusExp.clust2SpecClus[spCl][0]
        self.assertEqual(len(self.VclusExp.SpecClusters[clSymGroup]), 24)
        NumVecGroups = self.VclusExp.clus2LenVecClus[clSymGroup]
        self.assertEqual(NumVecGroups, 2)
        print(dx1, dx2)
        for vectorGroup, position in self.VclusExp.clust2vecClus[spCl]:
            self.assertEqual(len(self.VclusExp.vecVec[vectorGroup]), 24)
            print(self.VclusExp.vecVec[vectorGroup][position])

class test_FCC_orthogonal(test_FCC):
    def setUp(self):
        self.NSpec = 3
        self.Nvac = 1
        self.vacSpec = 2
        self.MaxOrder = 3
        a0 = 1
        self.a0 = a0
        self.crtype="FCC_orthogonal"
        basis_cube_fcc = [np.array([0, 0, 0]), np.array([0, 0.5, 0.5]), np.array([0.5, 0., 0.5]),
                          np.array([0.5, 0.5, 0.])]
        self.crys = crystal.Crystal(lattice=np.eye(3) * a0, basis=[basis_cube_fcc], chemistry=["A"], noreduce=True)
        self.N_units = 5
        self.superlatt = self.N_units * np.eye(3, dtype=int)
        self.superCell = supercell.ClusterSupercell(self.crys, self.superlatt)
        xVac = np.zeros(3)
        Rvac, civac = self.crys.cart2pos(xVac)
        self.vacsite = cluster.ClusterSite(civac, Rvac)
        self.vacsiteInd = self.superCell.index(Rvac, civac)[0]
        self.clusexp = cluster.makeclusters(self.crys, 1.01*a0, self.MaxOrder)

        numClusts = sum(len(clSet) for clSet in self.clusexp)
        print("Number of site clusters: {}".format(numClusts))

        self.VclusExp = Cluster_Expansion.VectorClusterExpansion(self.superCell, self.clusexp, self.NSpec,
                                                                 self.vacsite, self.vacSpec, self.MaxOrder)

        self.reqSites = None  #[i for i in range(10)]  # All interactions must have at least one of these sites
        self.VclusExp.generateSiteSpecInteracts(reqSites=self.reqSites)

        self.VclusExp.genVecClustBasis(self.VclusExp.SpecClusters)
        self.VclusExp.indexVclus2Clus()  # Index vector cluster list to cluster symmetry groups
        self.VclusExp.indexClustertoVecClus()

        self.Energies = np.random.rand(len(self.VclusExp.SpecClusters))
        print("Done setting up orthogonal FCC cluster expansion tests.")