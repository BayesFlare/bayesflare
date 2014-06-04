from ..misc import mkdir
import numpy as np
import bayesflare as pf

class Thresholder():
    """
    Class to carry out the thresholding proceedure in an automated manner.

    Parameters
    ----------
    B : ndarray
       An array of log-odds ratios.
    confidence : float, optional
       The threshold confidence (percentage). Defaults to 0.999 (i.e. 99.9%).

    
    """
    def __init__(self, B, confidence=0.999):
        self.B = B
        self.confidence = confidence
        self.folderpath = 'thresholding/'+str(B.identity_string())
        self.thrfolder = 'threshold/' + B.identity_string()
        
        self.calculate_threshold()

    def generate_threshold(self):
        """
        This function generates a histogram to calculate the lnB values at a given confidence level.
        The histogram is then stored in the ``threshold`` folder.
        """
        import pickle
        import glob
        import pylab as pl

        B = self.B

        files = glob.glob(self.folderpath+'/f*.pickle')
        filesp= glob.glob(self.folderpath+'/p*.pickle')
        data = np.array([]) 
        datap= np.array([])

        if len(files) < 1000 :
            print "[Note] The thresholding values must be calculated for this quarter, as they do not appear to exist. This can take several hours."
            repeats = 1000 - len(files)
            dt = B.lightcurve.dt()
            length = (B.lightcurve.cts[-1] - B.lightcurve.cts[0] )
            cadence = B.lightcurve.cadence
            pf.mkdir(self.folderpath)
            hours  = 3600
            M = pf.Flare(B.lightcurve.cts, amp=10)
            M.set_taus_gauss(0, 2*hours, 3)
            M.set_taus_exp(.5*hours, 5*hours, 30)
            P = pf.Transit(B.lightcurve.cts, amp=10)
            P.set_tauf(0.5*hours, 6*hours, 11)
            for i in np.arange(repeats-len(files)):
                print "[Note] Iteration ",  1+i, " of ", repeats, "(", ((float(i))/float(repeats))*100,"% completed)"
                A = pf.SimLightcurve(dt=dt, length=length, cadence=cadence)
                # Run the detection statistic for the flare model
                B = pf.Bayes(A, M)
                B.bayes_factors()                
                C = B.marginalise_full()
                D = pf.Bayes(A, P)
                D.bayes_factors()
                E = D.marginalise_full()
                F = C + E
                pickle.dump(F.lnBmargAmp, open(self.folderpath+'/f'+str(i)+'.pickle', 'wb'))
        for file in files:
            data = np.append(data,pickle.load(open(file, 'rb')))

        counts, bin_edges = np.histogram(data[np.isfinite(data)], bins=10000, density=True)
        cdf = np.cumsum(counts)/np.sum(counts)

        mkdir(self.thrfolder)
        pickle.dump(cdf, open(self.thrfolder+'/cdf.pickle', 'wb'))
        pickle.dump(bin_edges, open(self.thrfolder+'/binedge.pickle', 'wb'))

        print "[Note] The thresholding files have been found, and the histogram has been constructed"

    def calculate_threshold(self):
        """
        This function returns the lnB above which a value has a 99.9% confidence of being a flare
        as opposed to guassian noise.

        Parameters
        ----------
        quarter : int
           The *Kepler* quarter from which the data is taken
        cadence : string
           The cadence of the data, can be either 'long' or 'short'.
        confidence : float
           The level of confidence where the threshold should be drawn. 
           Default is 0.999

        Returns
        -------
        threshold : float
           the threshold ln(bayes factor) above which detections have ``confidence`` confidence of
           being a flare.
        
        """
        B = self.B
        confidence = self.confidence
        import pickle
        done = False
        try:
            self.cdf = pickle.load(open(self.thrfolder+'/cdf.pickle', 'rb'))
            self.bin_edges = pickle.load(open(self.thrfolder+'/binedge.pickle', 'rb'))
            thresh = np.where( self.cdf >= confidence )
            print "[Note] Now analysing the generated random noise to find flare threshold."
            self.threshold = self.bin_edges[thresh[0]][0]
        except IOError:
            # If the files containing the two histograms don't exist the must be created
            print "[Note] The files required to calculate the confidence levels were not found."
            self.generate_threshold()
            self.cdf = pickle.load(open(self.thrfolder+'/cdf.pickle', 'rb'))
            self.bin_edges = pickle.load(open(self.thrfolder+'/binedge.pickle', 'rb'))
            thresh = np.where( self.cdf >= confidence )
            print "[Note] Now analysing the generated random noise to find flare threshold."
            self.threshold = self.bin_edges[thresh[0]][0]

    def bayes_to_confidence(self, lnB):
        """
        This function returns a percentage confidence when given a lnB value.

        Parameters
        ----------
        quarter : int
           The *Kepler* quarter from which the data is taken
        cadence : string
           The cadence of the data, can be either 'long' or 'short'.
        lnB : float
           The ln(bayes factor) to be converted into a percentage conidence.

        Returns
        -------
        percentage : float
           The percentage confidence corresponding to ``lnB``
        """
        place = np.where( self.bin_edges >= lnB )
        place = place[0]-1
        if len(place) > 0:
            #print place
            place = place[0]
            return (self.cdf[place])
        else:
            return self.cdf[-1]
