import sys, os
import essentia.standard as ess
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import get_window
from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft
import codecs

def syllablesLoudnessHarmonics (seq, csvFile, audioFile):
    syllables, syllablesValues = csv2lists (csvFile)
    loudnessTable = loudnessHarmonics (audioFile)
    indexes = findSequence (seq, syllables)
    plotSyllablesLoudness (indexes, syllablesValues, loudnessTable)

#-------------------------------------------------------------------------------

def loudnessHarmonics (fileName, dumpFile = False):

    eps = np.finfo(np.float).eps
    
    path2SmsTools = '../../sms-tools-master'
    path2Models = os.path.join(path2SmsTools, 'software/models')
    sys.path.append(path2Models)
    
    import utilFunctions as UF
    import harmonicModel as HM
    import dftModel as DFT

    # Computing predominant melody
    H = 128
    M = 2048
    fs = 44100
    guessUnvoiced = True
    MELODIA = ess.PredominantMelody(guessUnvoiced=guessUnvoiced,
                                                   frameSize=M,
                                                   hopSize=H)
    audio = ess.MonoLoader(filename = fileName, sampleRate=fs) ()
    audioEL = ess.EqualLoudness() (audio)
    pitch = MELODIA(audioEL)[0]
    
    # Computing loudness including harmonics
    LOUDNESS = ess.Loudness()
    winAnalysis = 'hann'
    t = -80
    harmLoudness = []
    ## Synthesis
    nH = 15
    f0et = 5
    x = audioEL
    w = get_window(winAnalysis, M)
    hM1 = int(math.floor(w.size+1)/2)
    hM2 = int(math.floor(w.size/2))
    Ns = 4*H
    hNs = Ns/2
    startApp = max(hNs, hM1)
    pin = startApp
    pend = x.size - startApp
    x = np.append(np.zeros(startApp), x)
    x = np.append(x, np.zeros(startApp))
    N = 2 * M
    fftbuffer = np.zeros(N)
    yh = np.zeros(Ns)
    y = np.zeros(x.size)
    w = w / sum(w)
    sw = np.zeros(Ns)
    ow = triang(2 * H)
    sw[hNs-H:hNs+H] = ow
    bh = blackmanharris(Ns)
    bh = bh / sum(bh)
    sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
    hfreqp = []
    f0t = 0
    f0stable = 0
    cnt = 0
    while pin < pend:
        x1 = x[pin-hM1:pin+hM2]
        mX, pX = DFT.dftAnal(x1, w, N)
        ploc = UF.peakDetection(mX, t)
        iploc, ipmag, ipphase = UF. peakInterp(mX, pX, ploc)
        ipfreq = fs * iploc/N
        f0t = pitch[cnt]
        if ((f0stable == 0) & (f0t>0)
                or ((f0stable>0) & (np.abs(f0stable-f0t)<f0stable/5.0))):
            f0stable = f0t
        else:
            f0stable = 0
        hfreq, hmag, hphase = HM.harmonicDetection(ipfreq, ipmag, ipphase, f0t,
                                                   nH, hfreqp, fs)
        hfreqp = hfreq
        
        Yh = UF.genSpecSines(hfreq, hmag, hphase, Ns, fs)
        fftbuffer = np.real(ifft(Yh))
        yh[:hNs-1] = fftbuffer[hNs+1:]
        yh[hNs-1:] = fftbuffer[:hNs+1]
        yh_frame = sw*yh
        y[pin-hNs:pin+hNs] += yh_frame
        pin += H
        cnt+=1
        harmLoudness.append(LOUDNESS(yh_frame.tolist()))

    harmLoudness = np.array(harmLoudness)

    timeStamps = np.arange(harmLoudness.size) * H / float(fs)
    
    # Plotting
#    plt.plot(timeStamps, harmLoudness, color = 'b', linewidth=1)
#    plt.xlabel('Time (s)')
#    plt.ylabel('Amplitude')
#    plt.show()
    
    loudnessData = np.column_stack((timeStamps, harmLoudness))
    
    # Dumping a csv file
    if dumpFile:
        np.savetxt(fileName[:-4] + '-loudnessHarmonics.csv', loudnessData,
               delimiter=',')
               
    return loudnessData

#-------------------------------------------------------------------------------

def csv2lists (file):
    '''(path to file as str) -> list of unicode, numpy.array
    
    Given the path for the csv file, returns a list containing the unicode
    syllables from the firs column in file, and a numpy array containing the
    values in file, but replacing each syllable for its index in list.
    '''
    with codecs.open(file, 'r', encoding='UTF-16') as csvf:
        ulist = csvf.readlines()
    sylVal = []
    for i in ulist:
        sylVal.append([i.split(',')[0],
                        float(i.split(',')[1]),
                        float(i.split(',')[2])])
    syllables = [sylVal[i][0] for i in range(len(sylVal))]
    for i in range(len(sylVal)):
        sylVal[i][0] = i
    sylVal = np.array(sylVal)

    return syllables, sylVal
    
#-------------------------------------------------------------------------------

def findSequence(seq, lst):
    ''' (unicode, list of unicode) -> list of int
    
    Returns a list with the indexes for seq in lst.
    '''
    for i in range(len(lst)):
        if lst[i] == seq[0]:
            checklist = []
            indexes = []
            for j in range(len(seq)):
                checklist.append(lst[i+j] == seq[j])
                indexes.append(i+j)
            if False not in checklist:
                indexes = [(seq[k], indexes[k]) for k in range(len(seq))]
                return indexes
    return 'The sequence was not found in the list'
    
#-------------------------------------------------------------------------------

def plotSyllablesLoudness(indexes, syllablesValues, loudnessData,
                          normalization='None'):

    # Normalizing loudness
    timeStamps = loudnessData[:,0]
    loudnessValues = loudnessData[:,1]
    if normalization == 'None':
        loudnessValues = loudnessValues
        ymin = np.min(loudnessValues)
        ymax = np.max(loudnessValues)
    elif normalization == 'Mean':
        indSil = np.where(loudnessValues<=0)
        loudnessValues = loudnessValues / np.mean(loudnessValues)
        ymin = np.min(loudnessValues)
        ymax = np.max(loudnessValues)
    elif normalization == 'Max':
        loudnessValues = loudnessValues / float(np.max(loudnessValues))
        ymin = 0
        ymax = 1
    loudnessToPlot = np.column_stack((timeStamps, loudnessValues))
        

    sections = []
    for i in range(len(indexes)):
        startTime = syllablesValues[indexes[i][1], 1]
        endTime = syllablesValues[indexes[i][1], 2]
        indStart = np.argmin(abs(loudnessToPlot[:,0]-startTime))
        indEnd = np.argmin(abs(loudnessToPlot[:,0]-endTime))
        section = loudnessToPlot[indStart:indEnd,:]
        variance = np.var(section[:,1])
        sections.append((section, variance))
    
    fig = plt.figure()
    for i in range(len(sections)):
        ax = fig.add_subplot(int(str(len(sections)) + '1' + str(i+1)))
        ax.plot(sections[i][0][:,0], sections[i][0][:,1])
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.set_title(indexes[i][0], family='Droid Sans Fallback')
        ax.set_ylim((ymin, ymax))
        xini = sections[i][0][0,0]
        xend = sections[i][0][-1,0]
        ax.set_xlim(xini, xend)
        text = 'Variance=' + str(round(sections[i][1], 3))
        text_xpos = xini + ((xend - xini) * 0.05)
        ax.text(text_xpos, .8, text)
    plt.show()
    
    return sections
