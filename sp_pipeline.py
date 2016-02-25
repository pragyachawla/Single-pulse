#!/usr/bin/env python

"""
sp_pipeline.py

Make single pulse plots which include the waterfall plots and dedispersed time series with Zero-DM On/Off.
Also includes Signal-to-noise vs DM and DM vs Time subplots.
Usage on the command line:
./sp_pipeline.py --infile <any .inf file> --groupsfile <a groups.txt file> --mask <.rfifind.mask file> <psrfits file> <singlepulse files>

Takes fits 

Chitrang Patel - May. 21, 2015

Modified to sort candidates in order of increasing DM. Use psrfits_subband to subband the fits file into 128 subbands. Checks if 5*smearing time because of subbanding at the DM of the previous candidate is less than the pulse width of the current candidate. If yes, uses file subbanded at the previous candidate's DM. Otherwise, subbands at the DM of the current candidate. 

Pragya Chawla - December 9, 2015  
"""

import sys
import copy
from time import strftime
import infodata
from subprocess import Popen, PIPE, call
import numpy as np
import optparse
import waterfaller
import sp_utils
import bary_and_topo
import psr_utils
import rfifind
import show_spplots
#import memory
from sp_pulsar.formats import psrfits
from sp_pulsar.formats import spectra
from operator import itemgetter

DEBUG = True
def print_debug(msg):
    if DEBUG:
        print msg

def get_textfile(txtfile):
    """ Read in the groups.txt file.
    Contains information about the DM, time, box car width, signal to noise, sample number and rank    of groups. 
    """
    return  np.loadtxt(txtfile,dtype = 'str',delimiter='\n')

def group_info(rank, txtfile):
    """
    Extracts out relevant information from the groups.txt file as strings. 
    """
    files = get_textfile(txtfile)
    lis=np.where(files == '\tRank:             %i.000000'%rank)[0]#Checks for this contidion and gives its indices where true.
    # Extract the Max_ sigma value for the required parameters
    parameters=[]
    index=[]
    for i in range(len(lis)):
        temp_list = files[lis[i]-1].split()
        max_sigma = temp_list[2]
        max_sigma = float(max_sigma)
        max_sigma = '%.2f'%max_sigma
        # Extract the number of pulses for this group
        temp_list = files[lis[i]-6].split()
        number_of_pulses = int(temp_list[2])
        # Slice off a mini array to get the parameters from
        temp_lines = files[(lis[i]+1):(lis[i]+number_of_pulses+1)]
        # Get the parameters as strings containing the max_sigma
        parameters.append(temp_lines[np.array([max_sigma in line for line in temp_lines])])
	index.append(lis[i]) #adding position of candidate in file
    return parameters, index

def split_parameters(txtfile):
    """
    Splits the string into individual parameters and converts them into floats/int.
    **Function modified to generate a single list of all candidates ranked 4,5,6,7.  
    """
    files = get_textfile(txtfile)
    final_parameters=[]
    for group in [6, 5, 4, 3]:
	rank = group+1
	if files[group] != "Number of rank %i groups: 0 "%rank:
		print_debug(files[group])
		parameters, index = group_info(rank, txtfile)	
		for i in range(len(parameters)):
    		# If there is a degeneracy in max_sigma values, Picks the first one.(Can be updated to get the best pick) 
        		correct_values = parameters[i][0].split()
        		correct_values[0] = float(correct_values[0])
        		correct_values[1] = float(correct_values[1])
        		correct_values[1] = float('%.2f'%correct_values[1])
        		correct_values[2] = float(correct_values[2])
        		correct_values[3] = int(correct_values[3])
        		correct_values[4] = int(correct_values[4])
			correct_values.append(rank) #appended rank of candidate
			correct_values.append(index[i]) #appended position in txtfile
        		final_parameters.append(correct_values)
    return final_parameters

def topo_timeshift(bary_start_time, time_shift, topo):
    ind = np.where(topo == float(int(bary_start_time)/10*10))[0]
    return time_shift[ind]

def maskdata(data, start_bin, nbinsextra, maskfile):
    """
    Performs the masking on the raw data using the boolean array from get_mask.
    Inputs:
        data: raw data (psrfits object) 
        start_bin: the sample number where we want the waterfall plot window to start.
        nbinsextra: number of bins in the waterfall plot
    Output:
        data: 2D array after masking. 
    """
   
    if maskfile is not None:
        print 'masking'
        rfimask = rfifind.rfifind(maskfile)
        mask = waterfaller.get_mask(rfimask, start_bin, nbinsextra)
        # Mask data
        data = data.masked(mask, maskval='median-mid80')
    return data

def waterfall_array(start_bin, dmfac, duration, nbins, zerodm, nsub, subdm, dm, integrate_dm, downsamp, scaleindep, width_bins, rawdatafile, binratio, dat):
    """
    Runs the waterfaller. If dedispersing, there will be extra bins added to the 2D plot.
    Inputs:
        Inputs required for the waterfaller. dm, nbins, etc. 
    Outputs:
       data: 2D array as an "object" 
       array: 2D array ready to be plotted by sp_pgplot.plot_waterfall(array). 
    """
    data, bins = waterfaller.waterfall(start_bin, dmfac, duration, nbins, zerodm, nsub, subdm, dm, integrate_dm, downsamp, scaleindep, width_bins, rawdatafile, binratio, dat)
    array = np.array(data.data)
    if dm is not None:            # If dedispersing the data, extra bins will be added. We need to cut off the extra bins to get back the appropriate window size.   
        ragfac = float(nbins)/bins
        dmrange, trange = array.shape
        nbinlim = np.int(trange * ragfac)
    else:
        nbinlim = nbins
    array = array[..., :nbinlim]
    array = (array[::-1]).astype(np.float16)
    return data, array

def main():
    parser = optparse.OptionParser(prog="sp_pipeline..py", \
                        version=" Chitrang Patel (May. 12, 2015)", \
                        usage="%prog INFILE(PsrFits FILE, SINGLEPULSE FILES)", \
                        description="Create single pulse plots to show the " \
                                    "frequency sweeps of a single pulse,  " \
                    "DM vs time, and SNR vs DM,"\
                                    "in psrFits data.")
    parser.add_option('--infile', dest='infile', type='string', \
                        help="Give a .inf file to read the appropriate header information.")
    parser.add_option('--groupsfile', dest='txtfile', type='string', \
                        help="Give the groups.txt file to read in the groups information.") 
    parser.add_option('--mask', dest='maskfile', type='string', \
                        help="Mask file produced by rfifind. (Default: No Mask).", \
                        default=None)
    parser.add_option('-n', dest='maxnumcands', type='int', \
                        help="Maximum number of candidates to plot. (Default: 100).", \
                        default=100)
    options, args = parser.parse_args()
    if not hasattr(options, 'infile'):
        raise ValueError("A .inf file must be given on the command line! ") 
    if not hasattr(options, 'txtfile'):
        raise ValueError("The groups.txt file must be given on the command line! ") 
    
    files = get_textfile(options.txtfile)
    print_debug("Begining waterfaller... "+strftime("%Y-%m-%d %H:%M:%S"))
    if not args[0].endswith("fits"):
        raise ValueError("The first file must be a psrFits file! ") 
    print_debug('Maximum number of candidates to plot: %i'%options.maxnumcands)
    basename = args[0][:-5]
    filetype = "psrfits"
    inffile = options.infile
    topo, bary = bary_and_topo.bary_to_topo(inffile)
    time_shift = bary-topo
    inf = infodata.infodata(inffile)
    RA = inf.RA
    dec = inf.DEC
    MJD = inf.epoch
    mjd = Popen(["mjd2cal", "%f"%MJD], stdout=PIPE, stderr=PIPE)
    date, err = mjd.communicate()
    date = date.split()[2:5]
    telescope = inf.telescope
    N = inf.N
    numcands=0
    Total_observed_time = inf.dt *N
    print_debug('getting file..')
    values = split_parameters(options.txtfile)
    if len(values)> options.maxnumcands:
	values=sorted(values, key=itemgetter(5,1)) #sorting candidates based on ranks and snr
	values=values[-options.maxnumcands:] 
	print "More than", options.maxnumcands, "candidates, making plots for", options.maxnumcands, "candidates" 
    values = sorted(values, key=itemgetter(0))
    for ii in range(len(values)):
        #### Array for Plotting DM vs SNR
        print_debug("Making arrays for DM vs Signal to Noise...")
        temp_list = files[values[ii][6]-6].split()
        npulses = int(temp_list[2])
        temp_lines = files[(values[ii][6]+3):(values[ii][6]+npulses+1)]
        arr = np.split(temp_lines, len(temp_lines))
        dm_list = []
        time_list = []
        for i in range(len(arr)):
            dm_val= float(arr[i][0].split()[0])
            time_val = float(arr[i][0].split()[2])
            dm_list.append(dm_val)
            time_list.append(time_val)
        arr_2 = np.array([arr[i][0].split() for i in range(len(arr))], dtype = np.float32)
        dm_arr = np.array([arr_2[i][0] for i in range(len(arr))], dtype = np.float32)
        sigma_arr = np.array([arr_2[i][1] for i in range(len(arr))], dtype = np.float32)
	#### Array for Plotting DM vs Time is in show_spplots.plot(...)

                
        #### Setting variables up for the waterfall arrays.
        j = ii+1
        subdm = dm = sweep_dm= values[ii][0]
	sample_number = values[ii][3]
	rank=values[ii][5]
	width_bins = values[ii][4]
	#print "dm", dm 	
        #print "width_bins", width_bins 
	downsamp = np.round((values[ii][2]/sample_number/inf.dt)).astype('int')
	#print "downsamp", downsamp 
	pulse_width = width_bins*downsamp*inf.dt
	#print "pulse_width", pulse_width 
	if ii == 0:
	    mask_subband=rfifind.rfifind("%s_rfifind.mask"%(basename))
	    mask_subband.set_zap_chans(power=1000,plot=False)
	    mask_subband.set_weights_and_offsets()
	    mask_subband.write_weights(filename="%s_weights.txt"%(basename))
	    cmd="psrfits_subband -dm %.2f -nsub 128 -o %s_subband_%.2f -weights %s_weights.txt %s"%(dm,basename,dm,basename,args[0])
	    call(cmd, shell=True)
	    #subband args[0] at dm and then generate a file that will be set equal to rawdatafile
	    subband_file="%s_subband_%.2f_0001.fits" %(basename,dm)
	    dm_prev=dm
	    subband_prev= subband_file
	else:	
	    dm_diff=dm-dm_prev
	    t_smear=8.3e3*dm_diff*(350**-3)*(np.abs(rawdatafile.frequencies[0]-rawdatafile.frequencies[-1])/128.)
	    if (5*t_smear) > pulse_width:
		cmd="psrfits_subband -dm %.2f -nsub 128 -o %s_subband_%.2f -weights %s_weights.txt %s"%(dm,basename,dm,basename,args[0])
		call(cmd, shell=True)
		#subband args[0] at dm and then generate a file that will be set equal to rawdatafile
		subband_file="%s_subband_%.2f_0001.fits" %(basename,dm)
		dm_prev=dm
		subband_prev=subband_file

	rawdatafile = psrfits.PsrfitsFile(subband_file)
	bin_shift = np.round(time_shift/rawdatafile.tsamp).astype('int')
	integrate_dm = None
	sigma = values[ii][1]
        sweep_posn = 0.0
        bary_start_time = values[ii][2]
        topo_start_time = bary_start_time - topo_timeshift(bary_start_time, time_shift, topo)[0]
	binratio = 50
	scaleindep = False
        zerodm = None
        duration = binratio * width_bins * rawdatafile.tsamp * downsamp
	start = topo_start_time - (0.25 * duration)
	if (start<0.0):
            start = 0.0        
	if sigma <= 7:
	    nsub = 32
	elif sigma >= 7 and sigma < 10:
            nsub = 64
        else:
            nsub = 128
        nbins = np.round(duration/rawdatafile.tsamp).astype('int')
	start_bin = np.round(start/rawdatafile.tsamp).astype('int')
        dmfac = 4.15e3 * np.abs(1./rawdatafile.frequencies[0]**2 - 1./rawdatafile.frequencies[-1]**2)
	nbinsextra = np.round((duration + dmfac * dm)/rawdatafile.tsamp).astype('int')
	if (start_bin+nbinsextra) > N-1:
            nbinsextra = N-1-start_bin
	data = rawdatafile.get_spectra(start_bin, nbinsextra)
        data = maskdata(data, start_bin, nbinsextra, options.maskfile)
	#make an array to store header information for the .npz files
        temp_filename = basename+"_DM%.1f_%.1fs_rank_%i"%(subdm, topo_start_time, rank)
	# Array for Plotting Dedispersed waterfall plot - zerodm - OFF
        print_debug("Running waterfaller with Zero-DM OFF...")
        data, Data_dedisp_nozerodm = waterfall_array(start_bin, dmfac, duration, nbins, zerodm, nsub, subdm, dm, integrate_dm, downsamp, scaleindep, width_bins, rawdatafile, binratio, data)
        #Add additional information to the header information array
        text_array = np.array([subband_file, 'GBT', RA, dec, MJD, rank, nsub, nbins, subdm, sigma, sample_number, duration, width_bins, pulse_width, rawdatafile.tsamp, Total_observed_time, topo_start_time, data.starttime, data.dt, data.numspectra, data.freqs.min(), data.freqs.max()])
        #### Array for plotting Dedispersed waterfall plot zerodm - ON
        print_debug("Running Waterfaller with Zero-DM ON...")
	#print "before get_spectra",memory.resident()/(1024.0**3)
        data = rawdatafile.get_spectra(start_bin, nbinsextra)
	#print "after get_spectra",memory.resident()/(1024.0**3)
        data = maskdata(data, start_bin, nbinsextra, options.maskfile)
        zerodm = True
        data, Data_dedisp_zerodm = waterfall_array(start_bin, dmfac, duration, nbins, zerodm, nsub, subdm, dm, integrate_dm, downsamp, scaleindep, width_bins, rawdatafile, binratio, data)
        #print "waterfall",memory.resident()/(1024.0**3)
	####Sweeped without zerodm
        start = start + (0.25*duration)
        start_bin = np.round(start/rawdatafile.tsamp).astype('int')
        sweep_duration = 4.15e3 * np.abs(1./rawdatafile.frequencies[0]**2-1./rawdatafile.frequencies[-1]**2)*sweep_dm
        nbins = np.round(sweep_duration/(rawdatafile.tsamp)).astype('int')
        if ((nbins+start_bin)> (N-1)):
            nbins = N-1-start_bin
	#print "before get_spectra",memory.resident()/(1024.0**3)
        data = rawdatafile.get_spectra(start_bin, nbins)
	#print "after get_spectra",memory.resident()/(1024.0**3)
        data = maskdata(data, start_bin, nbins, options.maskfile)
        zerodm = None
        dm = None
        data, Data_nozerodm = waterfall_array(start_bin, dmfac, duration, nbins, zerodm, nsub, subdm, dm, integrate_dm, downsamp, scaleindep, width_bins, rawdatafile, binratio, data)
        #print "waterfall",memory.resident()/(1024.0**3)
	text_array = np.append(text_array, sweep_duration)
        text_array = np.append(text_array, data.starttime)
        text_array = np.append(text_array, bary_start_time)
        # Array to Construct the sweep
        if sweep_dm is not None:
            ddm = sweep_dm-data.dm
            delays = psr_utils.delay_from_DM(ddm, data.freqs)
            delays -= delays.min()
            delays_nozerodm = delays
            freqs_nozerodm = data.freqs
        # Sweeped with zerodm-on 
        zerodm = True
        downsamp_temp = 1
        data, Data_zerodm = waterfall_array(start_bin, dmfac, duration, nbins, zerodm, nsub, subdm, dm, integrate_dm, downsamp_temp, scaleindep, width_bins, rawdatafile, binratio, data)
        #print "waterfall",memory.resident()/(1024.0**3)
	# Saving the arrays into the .spd file.
        with open(temp_filename+".spd", 'wb') as f:
            np.savez_compressed(f, Data_dedisp_nozerodm = Data_dedisp_nozerodm.astype(np.float16), Data_dedisp_zerodm = Data_dedisp_zerodm.astype(np.float16), Data_nozerodm = Data_nozerodm.astype(np.float16), delays_nozerodm = delays_nozerodm, freqs_nozerodm = freqs_nozerodm, Data_zerodm = Data_zerodm.astype(np.float16), dm_arr= map(np.float16, dm_arr), sigma_arr = map(np.float16, sigma_arr), dm_list= map(np.float16, dm_list), time_list = map(np.float16, time_list), text_array = text_array)
        print_debug("Now plotting...")
        #print "Before plot..",memory.resident()/(1024.0**3)
	show_spplots.plot(temp_filename+".spd", args[1:], xwin=False, outfile = basename, tar = None)
        print_debug("Finished plot %i " %j+strftime("%Y-%m-%d %H:%M:%S"))
	#print "After plot..",memory.resident()/(1024.0**3)
	numcands+=1
        print_debug('Finished sp_candidate : %i'%numcands)
    print_debug("Finished running waterfaller... "+strftime("%Y-%m-%d %H:%M:%S"))


if __name__=='__main__':
    main()
