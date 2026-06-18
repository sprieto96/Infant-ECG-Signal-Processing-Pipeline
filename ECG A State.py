### ECG Pipeline for Sleep State Analysis - Sherbrooke Dataset - ACTIVE ###
### Import necessary libraries ###
import scipy.io as spio
import matplotlib.pyplot as plt
from tkinter import filedialog
import tkinter as tk
import numpy as np
import pandas as pd
import neurokit2 as nk

####### 1 Select .mat file  ##########
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select the .mat file")

if file_path:
    ##### 2. Load Numeric Data ###############
    mat_contents = spio.loadmat(file_path)
    
    ###### 3. Extract Variable Name #####
    var_name = [k for k in mat_contents if not k.startswith('__')][0]

    ###### 4. Transform Flatten to ECG Raw 1D Array #####
    ecg_raw = mat_contents[var_name].flatten()

    ##### 4a. Define file identity (edit once here) #####
    SUBJECT_ID = 'X'  # Edit this for each file to keep titles consistent across all plots and reports
    TITLE_PREFIX = f'{SUBJECT_ID}'

    ##### Print full recording info #####
    print(f"Total samples (full recording): {len(ecg_raw)}")
    print(f"Duration: {len(ecg_raw)/1000:.1f} seconds")
    print(f"Duration: {len(ecg_raw)/1000/60:.2f} minutes")

    ##### 5. Segment Signal: Slice to Sleep Window  ######
    start_sample = 0 * 1000        # = 0
    end_sample   = 0 * 1000     # = 300,000
    ecg_raw = ecg_raw[start_sample:end_sample] 

    ##### Print trimmed info #####
    print(f"Trimmed samples: {len(ecg_raw)}")
    print(f"Trimmed duration: {len(ecg_raw)/1000:.1f} seconds")
    print(f"Trimmed duration: {len(ecg_raw)/1000/60:.2f} minutes")

    ###### 6. Raw Signal Inspection: Plot a clean 10-second window ########################
    plt.figure(figsize=(12, 5))
    plt.plot(ecg_raw[:10000], color='blue', lw=0.8)
    
    ####### Style ######## 
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.title(f'{TITLE_PREFIX}:"A" Active Sleep State: Unfiltered Raw Signal from variable: {var_name}')
    plt.xlabel("Samples [N]")
    plt.ylabel("Amplitude [mV]")
    plt.show()

######### Raw Signal Inspection: Power Spectra Density #####################
######### Frequency Domain #########

psd_welch = nk.signal_psd(ecg_raw, method="welch", min_frequency=1, show=True)
plt.title(f'{TITLE_PREFIX}: "A" Active Sleep State: Power Spectral Density - Raw Unfiltered ECG Signal')
plt.xlim(0, 100)
plt.axvline(60, color='red', linestyle='--', label='60Hz Powerline')
plt.legend()
plt.show()

##### Filtered ##########
####### Apply Filters ######
####### Add lowcut=0.5 or 1.0 hz to fix the -70,000 offset #############
####### Add Hightcut to smooth baseline ############
####### Add Powerline = 60 #####################

ecg_filtered = nk.signal_filter(
    ecg_raw, 
    sampling_rate=1000, 
    lowcut=1, 
    highcut=40, 
    method='butterworth', 
    order=2, 
    powerline=60
)

####### Plotting Time ##########################
plt.figure(figsize=(12, 6))

## Zoom in for a cleaner look #####################
start, end = 4000, 6000 

plt.plot(ecg_filtered[start:end], color='red', label='Filtered (60Hz Powerline + Bandpass)')
plt.title(f'{TITLE_PREFIX}: "A" Active Sleep State: Filtered Signal: Powerline Noise & Baseline Centered')
plt.legend()
plt.xlabel("Samples [N]")
plt.ylabel("Voltage [mV]")
plt.show()


############## ECG Quality Check ####################
############# Possible that all beats exhibit high values (e.g. >0.95), indicative of consistent beat morphologies across the signal #################

quality = nk.ecg_quality(ecg_filtered, sampling_rate=1000, method="templatematch")
print(f"Quality score: {quality}")
plt.plot(quality)
plt.title(f'{TITLE_PREFIX}: "A" Active Sleep State: Quality Check')
plt.xlabel("Samples [N]")
plt.ylabel("Correlation Coefficient [r]")
X = 0.05 * (1.0 - np.min(quality))
lower = np.min(quality) - X
plt.ylim(lower, 1.0)


##### Print Quality Summary #########################
# Let's find the absolute minimum and the bottom 5 worst segments
min_val = np.min(quality)
# Use argpartition to find indices of the 5 lowest values efficiently
worst_indices = np.argpartition(quality, 5)[:5]
worst_values = quality[worst_indices]

print(f"--- Quality Report: ---")
print(f"Absolute Minimum Quality: {min_val:.4f}")
print(f"Absolute Maximum Quality: {np.max(quality):.4f}")
print(f"Indices of 5 lowest points: {worst_indices}")
print(f"Values of 5 lowest points: {worst_values}")
print(f"Mean Quality: {np.mean(quality):.4f}")
print(f"Standard Deviation: {np.std(quality):.4f}")
print(f"Median: {np.median(quality):.4f}")


# Calculate percentage of 'high quality' data (Threshold of 0.95 is standard)
threshold = 0.95
high_quality_pct = (np.sum(quality >= threshold) / len(quality)) * 100

print(f"--- File Reliability ---")
print(f"Percentage of file > {threshold} quality: {high_quality_pct:.2f}%")

if high_quality_pct > 95:
    print("Status: Excellent - Signal is highly reliable for analysis.")
elif high_quality_pct > 80:
    print("Status: Good - Majority of signal is usable.")
else:
    print("Status: Caution - Significant noise detected.")



############## Events Find / Epoch ##########################
sampling_rate = 1000
events = nk.events_find(quality, threshold=0.95, threshold_keep='above', duration_min=sampling_rate * 30)
onsets = events['onset']
offsets = events['onset'] + events['duration']
nk.events_plot([onsets, offsets], quality)
plt.legend(['Signal', 'Onset', 'Offset'])
plt.xlabel("Samples [N]")
plt.ylabel("Correlation Coefficient [r]")
plt.title(f'{TITLE_PREFIX}: "A" Active Sleep State: Detected High-Quality Segments (r > 0.95)')
plt.show()

###### Print Automated Segments Report #####################
print("\n--- Automated Segment Detection Report ---")
print(f"Total segments detected: {len(onsets)}")
print("-" * 50)

for i, (start, end) in enumerate(zip(onsets, offsets)):
    # convert to seconds 
    start_sec = start / sampling_rate
    end_sec = end / sampling_rate
    duration = (end - start) / sampling_rate
    
    print(f"Segment {i+1}:")
    print(f" Onset:  {start} samples ({int(start_sec // 60)}m {int(start_sec % 60)}s)")
    print(f" Offset: {end} samples ({int(end_sec // 60)}m {int(end_sec % 60)}s)")
    print(f" Duration: {duration:.2f} seconds")
    print("-" * 30)

#### Calculate Total Usable Yield #########################
total_usable_seconds = sum(events['duration']) / sampling_rate
print(f"\nTotal Usable Yield: {total_usable_seconds:.2f} seconds ({total_usable_seconds/60:.2f} minutes)")

################# Extract Peaks #####################
rpeaks, info = nk.ecg_peaks(ecg_filtered, sampling_rate=1000)

#### Plot R-Peaks #############################
nk.events_plot(info['ECG_R_Peaks'], ecg_filtered)
plt.title(f'{TITLE_PREFIX}: "A" Active Sleep State: R-Peak Validation')
plt.xlim(4000, 5000)
plt.xlabel("Samples [N]")
plt.ylabel("Voltage [mV]")
plt.show()



##### R-Peak Differences #####################################
rpeak_indices = np.where(rpeaks["ECG_R_Peaks"].values == 1)[0]

##### Check first 10 R Peaks ########################
print("First 10 R-peak indices:", rpeak_indices[:10])

##### Compute RR intervals in seconds ######################
rr_intervals_sec = np.diff(rpeak_indices) / 1000  # sampling rate = 1000 Hz

##### Plot RR intervals ##############################
plt.figure(figsize=(12,5))
plt.plot(rr_intervals_sec, marker='o', linestyle='-')
plt.title(f'{TITLE_PREFIX}: "A" Active Sleep State: RR Intervals from R-peaks')
plt.xlabel("Beat number")
plt.ylabel("RR interval (seconds)")
plt.ylim(0, 2)
plt.show()


###### Run peak correction ####################################
info, peaks_cleaned = nk.signal_fixpeaks(
    rpeaks, sampling_rate=1000, method="neurokit", iterative=True, show=True)
plt.show()



########## RR intervals from cleaned peaks #####################
rr_cleaned = np.diff(peaks_cleaned) / 1000  # convert samples to seconds
plt.figure(figsize=(12,5))
plt.plot(rr_cleaned, marker='o', linestyle='-', label='RR Interval')
plt.title(f'{TITLE_PREFIX}: "A" Active Sleep State: Cleaned RR Intervals (Correct Series)')
plt.xlabel("Beat number")
plt.ylabel("RR interval (s)")
plt.legend(loc='upper left')
plt.grid(True)

###### last 10 beats filtered ####################

# Make sure you have:
# ecg_filtered -> your filtered ECG signal
# peaks_cleaned -> cleaned R-peak indices from nk.signal_fixpeaks

# Select the last 10 R-peaks
last10_peaks = peaks_cleaned[-10:]

# Define a window around these peaks
# Take a bit before the first and after the last for context
start_idx = max(0, last10_peaks[0] - 200)   # 200 samples (~0.2 s) before
end_idx = min(len(ecg_filtered), last10_peaks[-1] + 200)  # 200 samples after

ecg_window = ecg_filtered[start_idx:end_idx]

# Adjust peak indices relative to the window
peaks_in_window = last10_peaks - start_idx

# Plot ECG with R-peaks marked
plt.figure(figsize=(12,5))
plt.plot(ecg_window, color='red', label='Filtered ECG')
plt.plot(peaks_in_window, ecg_window[peaks_in_window], 'bo', label='R-peaks', markersize=8)
plt.title(f'{TITLE_PREFIX}: "A" Active Sleep State: Filtered ECG - Last 10 R-peaks')
plt.xlabel("Samples [N]")
plt.ylabel("Amplitude [mV]")
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

############## EDR Function ####################
def compute_edr(ecg_filtered, peaks_cleaned, sampling_rate=1000):
    ecg_rate = nk.signal_rate(peaks_cleaned, sampling_rate=sampling_rate, desired_length=len(ecg_filtered))
    methods = ['vangent2019', 'soni2019', 'charlton2016', 'sarkar2015']
    fig, axes = plt.subplots(len(methods), 1, figsize=(14, 10), sharex=True)
    for i, method in enumerate(methods):
        edr = nk.ecg_rsp(ecg_rate, sampling_rate=sampling_rate, method=method)
        axes[i].plot(edr, lw=0.8, color='blue')
        axes[i].set_title(f"EDR - {method}")
        axes[i].set_ylabel("Amplitude [mV]")
        axes[i].grid(True)
    axes[-1].set_xlabel("Samples [N]")
    plt.suptitle(f'{TITLE_PREFIX}: "A" Active Sleep State: ECG-Derived Respiration (EDR)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

############## EDR ####################
compute_edr(ecg_filtered, peaks_cleaned, sampling_rate=1000)



############## ECG Delineation (P, Q, S, T waves) ####################
def compute_delineation(ecg_filtered, peaks_cleaned, sampling_rate=1000):
    
    # Delineate using dwt method (most precise for onsets/offsets)
    waves, signals = nk.ecg_delineate(
        ecg_filtered, 
        rpeaks=peaks_cleaned, 
        sampling_rate=sampling_rate, 
        method='dwt', 
        show=True, 
        show_type='all',
        check=True,
        window_start=-0.2, 
        window_end=0.2      
    )
     
     # Add title and axis labels after neurokit generates the plot
    plt.title(f'{TITLE_PREFIX}: "A" Active Sleep State: ECG Delineation - P, Q, R, S, T Waves')
    plt.xlabel("Time [Seconds]")
    plt.ylabel("Amplitude [mV]")
    plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    
    # Print summary of detected waves
    print("\n--- Wave Detection Summary ---")
    wave_keys = ['ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks', 
                 'ECG_P_Onsets', 'ECG_R_Onsets', 'ECG_R_Offsets', 'ECG_T_Offsets']
    for key in wave_keys:
        if key in waves:
            detected = np.sum(~np.isnan(waves[key]))
            print(f"{key}: {detected} detected")

    plt.show()
    
    return waves, signals

# Call it after peaks_cleaned is ready
waves, signals = compute_delineation(ecg_filtered, peaks_cleaned, sampling_rate=1000)

## ECG Phase function has bug waiting for update ####

############## ECG Rate ####################
def compute_ecg_rate(ecg_filtered, peaks_cleaned, sampling_rate=1000):
    
    # Compute heart rate interpolated over full signal length
    ecg_rate = nk.ecg_rate(
        peaks_cleaned,
        sampling_rate=sampling_rate,
        desired_length=len(ecg_filtered),
        interpolation_method='monotone_cubic',
        show=False
    )
    
    # Print summary
    print("\n--- Heart Rate Summary ---")
    print(f"Mean HR:  {ecg_rate.mean():.1f} bpm")
    print(f"Min HR:   {ecg_rate.min():.1f} bpm")
    print(f"Max HR:   {ecg_rate.max():.1f} bpm")
    print(f"Std HR:   {ecg_rate.std():.1f} bpm")

    # Plot
    plt.figure(figsize=(14, 5))
    plt.plot(ecg_rate, color='green', lw=0.8, label='Heart Rate (bpm)')
    plt.axhline(ecg_rate.mean(), color='red', linestyle='--', label=f'Mean HR: {ecg_rate.mean():.1f} bpm')
    plt.title(f'{TITLE_PREFIX}: "A" Active Sleep State: ECG Heart Rate Over Time')
    plt.xlabel("Samples [N]")
    plt.ylabel("Heart Rate (bpm)")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    return ecg_rate

# Call it after peaks_cleaned is ready
ecg_rate = compute_ecg_rate(ecg_filtered, peaks_cleaned, sampling_rate=1000)

# Convert R-peak indices to time in seconds for proper alignment
rpeak_times = rpeak_indices[1:] / 1000  # same length as rr_intervals_sec
sample_times = np.arange(len(ecg_rate)) / 1000  # ecg_rate in time (seconds)

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(rpeak_times, rr_intervals_sec, marker='o', linestyle='-', color='steelblue',
         alpha=0.7, markersize=4, label='Original RR Intervals (s)')
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("RR Interval (seconds)", color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_ylim(0, 2)

ax2 = ax1.twinx()
ax2.plot(sample_times, ecg_rate, color='crimson', lw=0.8, alpha=0.7, label='Heart Rate (bpm)')
ax2.set_ylabel("Heart Rate (bpm)", color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title(f'{TITLE_PREFIX}: "A" Active Sleep State: Original RR Intervals vs Interpolated Heart Rate')
plt.grid(True)
plt.tight_layout()
plt.show()

############## ECG Segmentation (Single Heartbeats) ####################
heartbeats = nk.ecg_segment(
    ecg_filtered,
    rpeaks=peaks_cleaned,
    sampling_rate=1000,
    show=True
)
plt.show()


##### Find HRV #########
hrv_indices = nk.hrv(rpeaks, sampling_rate=1000, show=True)



##### HRV Frequency Domain with extended PSD #####
hrv_freq = nk.hrv_frequency(
     rpeaks,
     sampling_rate=1000,
     show=True,
     vhf=(0.4, 2),         
     interpolation_rate=4      
     )


# ── 1. Let neurokit draw its default HRV figure ──────────────────────────────
hrv_indices = nk.hrv(rpeaks, sampling_rate=1000, show=True)

# Grab the figure neurokit just created (it's the current active figure)
fig = plt.gcf()
axes = fig.get_axes()

# axes layout from your screenshot:
#   axes[0] = RR distribution (top-left)
#   axes[1] = PSD (bottom-left)  <-- REPLACE THIS ONE
#   axes[2] = Poincaré main scatter (right)
#   axes[3] = Poincaré top marginal histogram
#   axes[4] = Poincaré right marginal histogram

ax_psd = axes[1]

# ── 2. Compute extended PSD manually ─────────────────────────────────────────
from scipy.signal import welch
from scipy.interpolate import interp1d

# RR intervals in ms and time axis in seconds
rri_ms   = np.diff(peaks_cleaned).astype(float)          # samples → ms at 1000 Hz
rri_time = np.cumsum(rri_ms) / 1000.0                    # seconds

# Interpolate at 4 Hz
interp_rate = 4
t_interp  = np.arange(rri_time[0], rri_time[-1], 1.0 / interp_rate)
rri_interp = np.interp(t_interp, rri_time, rri_ms)

freqs, psd = welch(rri_interp, fs=interp_rate, nperseg=256)

# Band masks — matching neurokit2 colours exactly
bands = {
    'ULF': ((freqs < 0.003),              'purple'),
    'VLF': ((freqs >= 0.003) & (freqs < 0.04),  'steelblue'),
    'LF':  ((freqs >= 0.04)  & (freqs < 0.15),  'green'),
    'HF':  ((freqs >= 0.15)  & (freqs < 0.4),   'orange'),
    'VHF': ((freqs >= 0.4)   & (freqs < 2.0),   'red'),
}

# ── 3. Clear the old PSD axes and redraw ─────────────────────────────────────
ax_psd.cla()

for label, (mask, colour) in bands.items():
    if mask.any():
        ax_psd.fill_between(freqs[mask], psd[mask],
                            color=colour, alpha=0.8, label=label)

ax_psd.set_xlim(0, 2.0)
ax_psd.set_ylim(bottom=0)
ax_psd.set_title('Power Spectral Density (PSD) for Frequency Domains')
ax_psd.set_xlabel('Frequency (Hz)')
ax_psd.set_ylabel('Spectrum (ms²/Hz)')
ax_psd.legend(loc='upper right', fontsize=8)
ax_psd.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()




#### Combined Summary Report Figure #########################

# Dynamic height calculation
n_data_rows = (
    3 +            # Recording Info
    8 +            # Quality Report
    3 +            # File Reliability
    2 +            # Segment Detection header + total
    len(onsets) +  # one row per segment
    1              # Total Usable Yield
)
fig_height = 1.0 + n_data_rows * 0.4

fig, ax = plt.subplots(figsize=(16, fig_height))
ax.axis('off')

# Determine status text
if high_quality_pct > 95:
    status = "Excellent - Signal is highly reliable for analysis."
elif high_quality_pct > 80:
    status = "Good - Majority of signal is usable."
else:
    status = "Caution - Significant noise detected."

# --- Build all rows ---
rows = []

# Section: Recording Info
rows.append(["RECORDING INFO", "", "#2c7bb6", True])
rows.append(["Trimmed Samples", f"{len(ecg_raw):,}", "#f0f4f8", False])
rows.append(["Trimmed Duration", f"{len(ecg_raw)/1000:.1f}s  /  {len(ecg_raw)/1000/60:.2f} min", "#f0f4f8", False])

# Section: Quality Report
rows.append(["QUALITY REPORT", "", "#2c7bb6", True])
rows.append(["Absolute Minimum Quality", f"{np.min(quality):.4f}", "#ffffff", False])
rows.append(["Absolute Maximum Quality", f"{np.max(quality):.4f}", "#ffffff", False])
rows.append(["Mean Quality", f"{np.mean(quality):.4f}", "#ffffff", False])
rows.append(["Median Quality", f"{np.median(quality):.4f}", "#ffffff", False])
rows.append(["Standard Deviation", f"{np.std(quality):.4f}", "#ffffff", False])
rows.append(["5 Lowest Indices", str(worst_indices), "#ffffff", False])
rows.append(["5 Lowest Values", str(np.round(worst_values, 4)), "#ffffff", False])

# Section: File Reliability
rows.append(["FILE RELIABILITY", "", "#2c7bb6", True])
rows.append([f"% of file > {threshold} quality", f"{high_quality_pct:.2f}%", "#d9ead3", False])
rows.append(["Status", status, "#d9ead3", True])

# Section: Automated Segment Detection
rows.append(["AUTOMATED SEGMENT DETECTION", "", "#2c7bb6", True])
rows.append(["Total Segments Detected", str(len(onsets)), "#f0f4f8", False])

for i, (start, end) in enumerate(zip(onsets, offsets)):
    start_sec = start / sampling_rate
    end_sec   = end / sampling_rate
    duration  = (end - start) / sampling_rate
    bg = "#ffffff" if i % 2 == 0 else "#f7f7f7"
    rows.append([
        f"Segment {i+1}",
        f"Onset: {start} ({int(start_sec//60)}m {int(start_sec%60)}s)  |  "
        f"Offset: {end} ({int(end_sec//60)}m {int(end_sec%60)}s)  |  "
        f"Duration: {duration:.2f}s",
        bg, False
    ])

# Section: Total Usable Yield
rows.append(["TOTAL USABLE YIELD",
             f"{total_usable_seconds:.2f}s  /  {total_usable_seconds/60:.2f} min",
             "#d9ead3", True])

# --- Render table ---
cell_text   = [[r[0], r[1]] for r in rows]
cell_colors = [[r[2], r[2]] for r in rows]

table = ax.table(
    cellText=cell_text,
    colLabels=["Metric / Section", "Value"],
    cellColours=cell_colors,
    loc='center',
    cellLoc='left',
    bbox=[0, 0, 1, 1]  # fills the full axes area
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.auto_set_column_width(col=[0, 1])

# Uniform row height across all cells
for (row, col), cell in table.get_celld().items():
    cell.set_height(1.0 / (len(rows) + 1))
    cell.PAD = 0.04

# Style column headers
for j in range(2):
    table[0, j].set_facecolor('#1a4a7a')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Bold section headers and yield row
for i, r in enumerate(rows):
    if r[3]:
        for j in range(2):
            cell = table[i + 1, j]
            cell.set_text_props(
                fontweight='bold',
                color='white' if r[2] == '#2c7bb6' else 'black'
            )

ax.set_title(
    f'{TITLE_PREFIX}: Full Signal & Quality Report',
    fontweight='bold', fontsize=13, pad=12
)

plt.savefig(f"{SUBJECT_ID}_quality_report.png", dpi=150, bbox_inches='tight')
plt.show()
