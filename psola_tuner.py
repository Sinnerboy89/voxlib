#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Use PSOLA to tune a speech/singing signal according to f0 file

import copy
import numpy as np
from scipy.io import wavfile
import crepe
from scipy.signal import resample
import matplotlib.pyplot as plt
from bisect import bisect_left

from frame import frame, fill_frame_data

WINDOW_SZ = 2
SRATE = 16000

class frame:
    # frame number as a string
    id = None
    # start of signal within frame (for example if frames overlap in a window)
    fstart = 0
    # size of signal in the frame
    fsz = 0.005
    # sample start position in the original waveform
    fstart_pos = None
    # lpc coeficients for the frame
    lpcs = None
    # order of lpcs (not including energy)
    lpcorder = None
    # pointer to previous frame
    prev_frame = None
    # pointer to next frame
    pst_frame = None
    # pointer to original float array of original waveform
    orig_wav = None
    # sample rate of data
    srate = None

def fill_frame_data(f):

    data = [0.] * -f.win_lb
    data += list(f.orig_wav[max(0, f.win_lb): min(len(f.orig_wav), f.win_ub)])
    data += [0.] * (f.win_ub - len(f.orig_wav))
    f.orig_data = np.array(data[:f.wsz])

def creepe(x, srate, fshift, threshold, min_f0=50, max_f0=1000):

    # compute f0s
    times, raw_f0s, conf, _ = crepe.predict(x, srate, verbose=True, viterbi=True)

    # limit f0s outside expected range
    f0s_1 = copy.copy(raw_f0s)
    f0s_1[f0s_1 > max_f0] = max_f0
    f0s_1[f0s_1 < min_f0] = min_f0

    # apply confidence threshold
    f0s_2 = copy.copy(f0s_1)
    f0s_2[conf < threshold] = 0.
    
    # lerp across "zero" regions
    total_frames = len(raw_f0s)
    t_axis = np.linspace(0, fshift, total_frames)
    f0s_3 = copy.copy(f0s_2)
    f0s_3[f0s_3 == 0.] = np.interp(t_axis[f0s_3 == 0.], t_axis[f0s_3 != 0.], f0s_2[f0s_3 != 0.])
    
    return f0s_3

def f0s_to_frames(f0s, fshift, x, srate):
    
    frame_times = []
    i = 0
    ft_s = 0
    while i < len(f0s):
        ft_e = ft_s + (1 / f0s[i])
        frame_times.append([ft_s, ft_e])
        i = int(ft_e / fshift)
        ft_s = ft_e

    # ensure start is covered
    if frame_times[0][0]:
        frame_times.insert(0, [0, frame_times[0][0]])
        
    # ensure end is covered
    if frame_times[-1][1] != len(x)/srate:
        frame_times.append([frame_times[-1][1], len(x)/srate])
    
    # generate frame objects
    frames = gen_frames(frame_times, x, srate)
    
    return frames

def window_frames(frames):
    
    for f in frames:

        print(f"{f.id} of {len(frames)}", end="\r")
        
        # window length and position
        f.win_cen = int(f.fstart_pos + int(f.fsz/2))
        f.win_lb = int(f.win_cen - (f.fsz * WINDOW_SZ * 0.5)) 
        f.win_ub = int(f.win_cen + (f.fsz * WINDOW_SZ * 0.5))
        f.wsz = f.win_ub - f.win_lb
        f.fstart = f.win_lb - f.fstart_pos
        
        # windowed data
        hsz = f.wsz
        fill_frame_data(f)
        off = int((hsz - f.wsz)/2)
        if off >= 0:
            f.hanning_window = np.hanning(hsz)[off:off+f.wsz]
        else:
            f.hanning_window = np.array(([0.] * -off) + list(np.hanning(hsz)) + ([0.] * -off))
        f.windowed_data = np.multiply(f.hanning_window, f.orig_data)

def framework_transfer(frames_src, frames_target):

    fa_mps = []
    for fa in frames_src:
        fa_mps.append(fa.midpoint)
    for ft in frames_target:
        print(f"{ft.id} of {len(frames_target)}", end="\r")
        # select closest grain (via midpoint)
        fa_idx = bisect_left(fa_mps, ft.midpoint) - 1
        selected_fa = frames_src[fa_idx]
        # update target frame (positioning comes from target, size and signal comes from grain)
        ft.fsz = selected_fa.fsz
        ft.wsz = selected_fa.wsz
        ft.win_cen = int(ft.fstart_pos + int(ft.fsz/2))
        ft.win_lb = int(ft.win_cen - (ft.fsz * WINDOW_SZ * 0.5)) 
        ft.windowed_data = selected_fa.windowed_data
        
def render(frames):
    
    # initialise output audio array
    audio = array.array('f')
    audio.fromlist([0.0] * (frames[-1].fstart_pos + frames[-1].fsz))
    f = frames[0]
    
    for f in frames:
        print(f"{f.id} of {len(frames)}", end="\r")
        for i in range(f.wsz):
            pos = i + f.win_lb
            if pos > 0 and pos < len(audio):
                audio[pos] = audio[pos] + f.windowed_data[i]
                
    return audio
        
def main():       
    # Setup option parsing
    from optparse import OptionParser
    usage="usage: %prog [options] <input wav>\n" \
        "e.g. python tune_psola.py vde_z0001_020.wav" \
        "Outputs modified wav" 
    parser = OptionParser(usage=usage)
    parser.add_option("-o", "--outfp", help="output filepath")
    parser.add_option("-Q", "--targetfp", help="target wav or f0 filepath")
    parser.add_option("-f", "--fshift", default=0.01, help="fshift for f0s")
    parser.add_option("-t", "--threshold", default=0.8, help="threshold for CREPE voicing")
    parser.add_option("-m", "--min_f0", default=50, type="float", help="minimum expected f0")
    parser.add_option("-M", "--max_f0", default=1000, type="float", help="maximum expected f0")
    opts, args = parser.parse_args()
    
    print("Reading in LPCVoc and target audio...")
    srate_lpcvoc, audio_lpcvoc = wavfile.read(args[0])
    srate_target, audio_target = wavfile.read(opts.targetfp)
    if srate_target != 16000:
        print(f"WARNING: target audio sample rate isn't 16kHz (it is {srate_target}); CREPE will resample to 16kHz")
    
    print("Running CREPE over both...")
    f0s_2_lpcvoc = creepe(audio_lpcvoc, SRATE, opts.fshift, opts.threshold, min_f0=opts.min_f0, max_f0=opts.max_f0)
    f0s_2_target = creepe(audio_target, srate_target, opts.fshift, opts.threshold, min_f0=opts.min_f0, max_f0=opts.max_f0)
    
    # save/load crepe f0s to/from file
    #np.savetxt("/home/chrisb/test_lpcvoc.f0", f0s_2_lpcvoc)
    #np.savetxt("/home/chrisb/test_target.f0", f0s_2_target)
    #f0s_2_lpcvoc = np.loadtxt("/home/chrisb/test_lpcvoc.f0")
    #f0s_2_target = np.loadtxt("/home/chrisb/test_target.f0")
    
    # plots
    #plt.subplot(2, 1, 1)
    #plt.plot(f0s_2_lpcvoc)
    #plt.plot(f0s_2_target)
    #plt.legend(["lpcvoc", "target"])
    #plt.title('f0s')
    #plt.show()
    
    print("Converting LPCVoc f0s to pitch-synchronous framework...")
    frames_lpcvoc = f0s_to_frames(f0s_2_lpcvoc, opts.fshift, audio_lpcvoc, SRATE)
    print("Converting target f0s to pitch-synchronous framework...")
    frames_target = f0s_to_frames(f0s_2_target, opts.fshift, audio_lpcvoc, SRATE)
    
    print("Extracting analysis 'grains'...")
    window_frames(frames_lpcvoc)
    print("Extraction done")
    
    print("PSOLA-like transferring of 'grains' to target framework...")
    framework_transfer(frames_lpcvoc, frames_target)
    print("Transfer done ")

    print("Rendering 'grains' with target framework...")
    y = render(frames_target)
    print("Rendering done")

    print("Saving tuned output to .wav...")
    wavfile.write(opts.outfp, SRATE, np.array(y).astype("int16"))
    
    print("All done!")
        
if __name__ == "__main__":
    main()
