#!/usr/bin/env python

"""
Script to run detection for Kepler Q1 pulsars
"""

# use API to get list of files (greatly inspired by kplr python API) http://dan.iel.fm/kplr/
import json
import urllib
import urllib2
import os
import sys
import datetime
import numpy as np
import bayesflare as bf
from copy import copy

from optparse import OptionParser

__version__= "1.0"

class Adapter(object):
  """
  An :class:`Adapter` is a callable that maps a dictionary to another
  dictionary with different keys and specified data types. Missing/invalid
  values will be mapped to ``None``.

  This is based on the Adapter class of the kplr python API http://dan.iel.fm/kplr/

  :param parameters:
     A dictionary of mappers. The keys should be the keys that will be in
    the input dictionary and the values should be 2-tuples with the output
    key and the callable type converter.

  """

  def __init__(self, parameters):
    self._parameters = parameters

  def __call__(self, row):
    row = dict(row)
    final = {}
    for longname, (shortname, conv) in self._parameters.items():
      try:
        final[shortname] = conv(row.pop(longname, None))
      except (ValueError, TypeError):
        final[shortname] = None

    return final

# main function
if __name__=='__main__':

  usage = "Usage: %prog [options]"

  description = \
  """
     Search for flare in Kepler data based on a Bayesian odds ratio detection statistic. This \
is hardcoded to only use long cadence data. Various vetos will also be applied on certain \
stellar characteristics. The defaults are consistent with the vetoes used in Pitkin, \
Williams, Fletcher and Grant (2014).
  """

  parser = OptionParser( usage = usage, description = description, version = __version__ )

  parser.add_option("-L", "--kepler-dir", dest="filedir",
                    help="A directory of Kepler data files.")

  parser.add_option("-T", "--threshold", dest="threshold",
                    help="Odds ratios above this value will count as detections [default: %default].",
                    type="float", default=16.5)

  parser.add_option("-E", "--Teff", dest="Teff",
                    help="The acceptance condition on stellar effective temperature to include in search [default: %default],",
                    default="<=5150")

  parser.add_option("-l", "--logg", dest="logg",
                    help="The acceptance condition on stellar log(surface gravity) to include in search [default: %default],",
                    default=">=4.2")

  parser.add_option("-Q", "--quarter", dest="quarter",
                    help="The Kepler Quarter to include in search [default: %default],",
                    type="int", default=1)

  parser.add_option("-o", "--output-file", dest="outfile",
                    help="An output file for the results (they will be in JSON format).")

  # read in arguments
  (opts, args) = parser.parse_args()

  # check output file is given
  if not opts.__dict__['outfile']:
    print "Error... no output file specified!"
    sys.exit(0)
  else:
    outfile = opts.outfile

  print """
If file %s exists then the analysis will restart with the final star \
in that files and append the data. If you want to restart from scratch \
then delete the file.
      """ % outfile

  # local directory containing kepler data
  if not opts.__dict__['filedir']:
    print "Error... no Kepler data directory specified!"
    sys.exit(0)
  else:
    datadir = opts.filedir

  # set generic params
  params = {}
  params["action"] = "Search"
  params["outputformat"] = "JSON" # output in JSON format
  params["coordformat"] = "dec" # output format for RA and DEC http://archive.stsci.edu/kepler/help/search_help.html#coordf
  params["verb"] = 3
  params["max_records"]= 50001 # get all quarter 1 stars (~ 23000 stars)
  #params["max_records"]= #10
  params["ordercolumn1"] = "ktc_kepler_id" # sort by ascending kepler ID
  # params["ordercolumn1"] = "kic_teff" # example of how you might sort by Teff

  # set required search criterion
  params["kic_teff"] = opts.Teff              # Teff condition
  params["kic_logg"] = opts.logg              # log(g) condition
  params["ktc_target_type"] = "LC"            # long cadence
  params["sci_data_quarter"] = opts.quarter   # quarter

  # params["ktc_kepler_id"] = ">=1873543" # example to use to specify range of Kepler IDs

  # MAST url
  mast_url = "http://archive.stsci.edu/kepler/{0}/search.php"

  r = urllib2.Request(mast_url.format("data_search"), data=urllib.urlencode(params))
  handler = urllib2.urlopen(r) # get the data

  code = handler.getcode()
  if code != 200:
    print "Error... problem getting data."

  txt = handler.read()
  results = json.loads(txt) # parse into JSON format

  kid_adapter = Adapter({"Kepler ID": ("kepid", int), # get kepler IDs
                         "Condition flag": ("condition_flag", unicode), # get condition
                         "Teff": ("teff", int), # get teff
                         "Log G": ("logg", float)}) # get Log G

  # select the Kepler ID and condition flag from the output
  kidspreveto = [kid_adapter(row) for row in results]

  Ntot = len(kidspreveto) # total number of stars

  # veto anything with one of the Conditions http://archive.stsci.edu/kepler/condition_flag.html
  # This should remove exoplanets, KOI (planetary candidates), eclipsing binaries etc.
  kids = [v for v in kidspreveto if v["condition_flag"] == 'None']

  # Remove any extra stars from the eclipsing binary list from Prsa, A. et al, AJ, 141 (2011)
  # http://arxiv.org/abs/1006.2815 (see table at
  # http://iopscience.iop.org/1538-3881/141/3/83/suppdata/aj376947t1_mrt.txt).
  # Not all of these are removed by the condition flag(!)
  ebtable = 'http://iopscience.iop.org/1538-3881/141/3/83/suppdata/aj376947t1_mrt.txt'
  Npreebveto = len(kids)
  try:
    f = urllib.urlopen(ebtable)
    ebs = [int(v.split()[0]) for v in f.readlines()[45:]] # ignore header lines
    f.close()

    kidstmp = [v for v in kids if v["kepid"] not in ebs] # add stars not in the list

    kids = kidstmp
  except:
    print "Error... could not get Kepler eclipsing binary table"
    sys.exit(0)

  print "%d stars removed by eclipsing binary veto" % (Npreebveto - len(kids))

  # REMOVE ANY STARS WITH PERIODS OF LESS THAN A GIVEN THRESHOLD

  # URL of table 1 from McQuillan, A. et al (2014) ApJS, 211, 24 http://arxiv.org/abs/1402.5694
  periodtable = "http://iopscience.iop.org/0067-0049/211/2/24/fulltext/apjs492452t1_mrt.txt"
  try:
    f = urllib.urlopen(periodtable)
    pt = f.readlines()
    f.close()

    # ignore the 32 header lines and extract the KIC and period (days)
    kp = [(int((v.split())[0]), float((v.split())[4])) for v in pt[32:]]
  except:
    # try alternative URL
    periodtable = "http://arxiv.org/src/1402.5694v2/anc/Table_1_Periodic.txt"

    try:
      f = urllib.urlopen(periodtable)
      pt = f.readlines()
      f.close()

      # ignore the one header line and extract the KIC and period (days)
      kp = [(int((v.split(','))[0]), float((v.split(','))[4])) for v in pt[1:]]
    except:
      print "Error... could not get Kepler period table"
      sys.exit(0)

  # URL of table 2 from McQuillan, A. et al (2014) ApJS, 211, 24 http://arxiv.org/abs/1402.5694
  periodtable2 = "http://arxiv.org/src/1402.5694v2/anc/Table_2_Non_Periodic.txt"
  try:
    f = urllib.urlopen(periodtable2)
    pt = f.readlines()
    f.close()

    # ignore the one header line and extract the period values that are not NaN (append to previoud list)
    kp += [(int((v.split(','))[0]), float((v.split(','))[4])) for v in pt[1:] if (v.split(','))[4] != 'nan']
  except:
    print "Error... could not get Kepler period table 2"
    sys.exit(0)

  # URL of table from Reinhold, T. et al (2013) A&A, 560 http://arxiv.org/abs/1308.1508
  periodtable3 = "http://cdsarc.u-strasbg.fr/vizier/ftp/cats/J/A+A/560/A4/table.dat"
  # (see README at http://cdsarc.u-strasbg.fr/vizier/ftp/cats/J/A+A/560/A4/ReadMe)
  # this table contains primary and secondary periods and we will veto both
  try:
    f = urllib.urlopen(periodtable3)
    pt = f.readlines()
    f.close()

    # first get primary periods (no header lines to ignore)
    kp += [(int((v.split())[0]), float((v.split())[1])) for v in pt]

    # now get secondary periods (ignore cases where no period is given [marked by a '---'])
    kp += [(int((v.split())[0]), float((v.split())[3])) for v in pt if (v.split())[3] != '---']
  except:
    print "Error... could not get third Kepler period table"
    sys.exit(0)

  periodlim = 2.0 # 2 day period limit threshold

  Npreperiodveto = len(kids) # number of stars pre- the period veto

  kidstmp = []
  for kic in kids:
    periodveto = False
    for p in kp: # loop through period data
      if kic["kepid"] == p[0]:
        if p[1] < periodlim:
          periodveto = True # veto this star
        break
    if not periodveto: # only add stars that are not vetoed
      kidstmp.append(kic)

  kids = kidstmp

  Npostperiodveto = len(kids) # length of final dataset

  # list files in datadir
  kfilestmp = os.listdir(datadir)

  kfiles = []
  # run through kids get the files for them
  kidstmp = []

  for kid in kids:
    kidn = 'kplr%09d' % kid["kepid"]

    for f in kfilestmp:
      if kidn in f:
        if 'slc' not in f: # veto short cadence files
          kfiles.append(os.path.join(datadir, f))
          kidstmp.append(kid)
          break

  kids = kidstmp

  Nfinal = len(kids)

  threshold = opts.threshold

  outdict = {} # a dictionary to contain the output json file

  if os.path.isfile(outfile):
    # try opening file with json
    try:
      f = open(outfile, 'r')

      # load data currently in file
      try:
        outdict = json.load(f)
        f.close()
      except:
        print "Error... could not open output file. File will be overwritten during analysis"
    except:
      print "Error... could not open output file. File will be overwritten during analysis"


  # create dictionary of output data
  if len(outdict) == 0:
    outdict = {
      "Analysis": "Kepler flare search",
      "Analyser": " ", # name of person running analysis
      "Notes":    " ", # and notes
      "Condition flag requirement": "None", # when gathering data this is the required condition flag
      "Teff requirement": params["kic_teff"],
      "log(g) requirement": params["kic_logg"],
      "Cadence requirement": params["ktc_target_type"],
      "Kepler quarter": params["sci_data_quarter"],
      "Total no. of stars": Ntot,
      "No. stars post-condition flag veto": Npreperiodveto,
      "No. stars post-period veto": Npostperiodveto,
      "No. stars analysed": 0,
      "Period limit": periodlim, # lower limit on stellar period
      "log odds ratio threshold": threshold, # threshold for flare detection
      "Star list": [], # list of stars being analysed
      "Flaring stars": [], # list of stars with detected flares
      "Flaring star data": {}, # dictionary of dictionaries for flare data
      "No. flares": 0, # number of flares found
      "Rejected stars": [] # list of rejected stars (i.e. with data gaps)
    }

    starlist = None
    totflares = 0
    nf = 0
  else:
    # try getting star list
    try:
      starlist = outdict["Star list"]
      totflares = outdict["No. flares"]
      nf = outdict["No. stars analysed"]
    except:
      print "No star list in file. Analysis will start from beginning."
      outdict["Star list"] = []
      outdict["Star data"] = []
      outdict["Flaring stars"] = []
      outdict["Flaring star data"] = {}
      outdict["No. flares"] = 0
      outdict["Rejected stars"] = []
      totflares = 0

  # update date and time of analysis
  now = datetime.datetime.utcnow()
  outdict["Date"] = now.isoformat(' ')

  expand = 1 # expand each detected flare segment by 1 on either end (this will help merge adjacent detections from the same flare)
  maxgap = 2 # the maximum allowed gap in the data

  # run through files and perform analyses
  for i, kfile in enumerate(kfiles):
    # every 100 iterations rewrite the output file
    if i % 100:
      # dump current data
      try:
        f = open(outfile, 'w')
        json.dump(outdict, f, indent=2)
        f.close()
      except:
        print "Error... problem outputting JSON data file"
        sys.exit(0)

    print "Iteration %d: KIC %d" % (i+1, kids[i]["kepid"])

    flarelc = bf.Lightcurve(curve=kfile, maxgap=maxgap)
    # check if the data had gaps
    if flarelc.datagap:
      print "KIC %d had a data gap greater than %d. Ignoring this star." % (kids[i]["kepid"], maxgap)
      outdict["Rejected stars"].append(kids[i]["kepid"])
      continue

    # check if star had already been analysed
    if starlist != None:
      if kids[i]["kepid"] in starlist:
        print "KIC %d has already been analysed" % kids[i]["kepid"]
        continue

    nf = nf + 1
    outdict["No. stars analysed"] = nf # update number of stars
    outdict["Star list"].append(kids[i]["kepid"]) # append to star list

    # calculate odds ratio
    odds = bf.OddsRatioDetector(flarelc, noiseestmethod='tailveto', tvsigma=1.0)
    odds.set_noise_impulse(noiseimpulseparams={'t0': (0, (odds.bglen-1.)*flarelc.dt(), odds.bglen)})

    lnO, ts = odds.oddsratio()

    # find flares
    flarelist, Nflares, maxlist = odds.thresholder(lnO, threshold, expand=expand)

    if i == 0:
      # set some analysis data
      outdict["Noise estimation method"] = odds.noiseestmethod # method for noise estimation
      outdict["Running window length"] = odds.bglen # number of time bins used for running window
      outdict["Polynomial background order"] = odds.bgorder # polynomial order
      outdict["Flare taug"] = odds.flareparams['taugauss'] # range in taug
      outdict["Flare taue"] = odds.flareparams['tauexp']   # range in taue
      outdict["Amplitude priors"] = odds.flareparams['amp']

    # write out flare info for star
    if Nflares > 0:
      print "Found %d flares for star KIC %d" % (Nflares, kids[i]["kepid"])

      totflares = totflares + Nflares
      outdict["No. flares"] = totflares
      outdict["Flaring stars"].append(kids[i]["kepid"])

      # create lists of flare data
      maxL = []
      maxIdx = []
      maxT = []
      for (maxlno, maxidx) in maxlist:
        maxL.append(maxlno)     # maximum odds ratio for each flare candidate
        maxIdx.append(maxidx)   # index of maximum
        maxT.append(ts[maxidx]) # time of maximum

      segodds = []   # odds ratios for all above threshold segments
      segtimes = []  # times of all above threshold segments
      segidxs = []   # indices of all above threshold segments
      for (starts, ends) in flarelist:
        sidxs = np.arange(starts, ends)
        segidxs.append(sidxs.tolist())
        segtimes.append(ts[sidxs].tolist())
        segodds.append(np.copy(lnO)[sidxs].tolist())

      # get the noise estimate
      tmpcurve = copy(flarelc)
      tmpcurve.detrend(method='savitzkygolay', odds.bglen, odds.bgorder)
      if odds.noiseestmethod == 'powerspectrum':
        sk = bf.estimate_noise_ps(tmpcurve, estfrac=odds.psestfrac)[0]
      elif odds.noiseestmethod == 'tailveto':
        sk = bf.estimate_noise_tv(tmpcurve.clc, sigma=odds.tvsigma)[0]
      else:
        print "Error... Noise estimation method must be 'powerspectrum' or 'tailveto'"
        sys.exit(0)

      datadict = {
        "Kepler ID": kids[i]["kepid"],
        "No. flares": Nflares,
        "Teff": kids[i]["teff"],
        "log(g)": kids[i]["logg"],
        "Max. odds ratios": maxL,
        "Max. indices": maxIdx,
        "Max. times": maxT,
        "Segment odds ratios": segodds,
        "Segment indices": segidxs,
        "Segment times": segtimes,
        "Noise sigma": sk, # estimate of the noise standard deviation
        "Lightcurve DC background": flarelc.dc # DC (median) background of the lightcurve
      }

      # add data to dictionary
      outdict["Flaring star data"]["KPLR%09d" % (kids[i]["kepid"])] = datadict

  # output final data
  try:
    f = open(outfile, 'w')
    json.dump(outdict, f, indent=2)
    f.close()
  except:
    print "Error... problem outputting JSON data file"
    sys.exit(0)
