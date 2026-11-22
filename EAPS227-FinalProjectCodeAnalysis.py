# Imports
import pandas as pd
from metpy.calc import saturation_vapor_pressure
from metpy.calc import mixing_ratio
from metpy.units import units
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# How to Run:
# 
# Download both csv files and alter the read_csv
# lines below to match the location. Ensure all 
# of the above are installed (pip is reccomended
# as all of the code was written in python).
#
# Kestrel A is grass, Kestrel B is concrete
#
# NOTE: Kestrel A did data collection every minute,
# wheras Kestrel B collected every 10 seconds and had
# extra data packed into the csv
#
# ## = section, # = subsection
#
# See the README for further details. 


## Constants

# Locations (user can change if needed)
kestrel_A_path = 'HEAT - 2644217 Kestrel session export.csv'
kestrel_B__path = 'HEAT_-_2644237_Oct_29_2026_12_25_30_PM.csv'
graph_name_output = 'vapor_pressure_comparison.png'
txt_file_output = 'EAPS227_Analysis_Results.txt'

# Sunrise Time (in EDT) <- notes for user
sunrise_time = 753
cloud_cover_end = 930


## Read and clean datasets, then combine them to align time values
df_A = pd.read_csv(kestrel_A_path, header=3, skiprows=[4])
df_B = pd.read_csv(kestrel_B__path, header=3, skiprows=[4])

df_clean_A = df_A.dropna(subset=['FORMATTED DATE_TIME', 'Dew Point', 'Temperature']).copy()
df_clean_A['datetime'] = (
    pd.to_datetime(df_clean_A['FORMATTED DATE_TIME'], errors='coerce')
    .dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='shift_forward')
    .dt.tz_convert('UTC')
)
df_clean_A = df_clean_A.sort_values('datetime')

df_clean_B = df_B.dropna(subset=['FORMATTED DATE_TIME', 'Dew Point', 'Temperature']).copy()
df_clean_B['datetime'] = (
    pd.to_datetime(df_clean_B['FORMATTED DATE_TIME'], errors='coerce')
    .dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='shift_forward')
    .dt.tz_convert('UTC')
)
df_clean_B = df_clean_B.sort_values('datetime')
df_clean_B = df_clean_B[df_clean_B['datetime'] >= '2026-10-29']
df_clean_B = df_clean_B[df_clean_B['datetime'] <= '2026-10-29 16:17:00']
# df_clean_B = df_clean_B[df_clean_B['datetime'] >= '2026-10-29']
# df_clean_B = df_clean_B[df_clean_B['datetime'] <= '2026-10-29 12:17:00']


df_A_indexed = df_clean_A.drop_duplicates(subset=['datetime']).set_index('datetime')[['Temperature', 'Dew Point']]

df_B_indexed = df_clean_B.drop_duplicates(subset=['datetime']).set_index('datetime')[['Temperature', 'Dew Point']]
df_B_resampled = df_B_indexed.resample('1min').mean()

combined_df = pd.DataFrame({
    'Temp_Grass': df_A_indexed['Temperature'],
    'Dewpoint_Grass': df_A_indexed['Dew Point'],
    'Temp_Concrete': df_B_resampled['Temperature'],
    'Dewpoint_Concrete': df_B_resampled['Dew Point']
})

combined_df = combined_df.interpolate(method='linear')
combined_clean_df = combined_df.dropna()

# Get Vapor Pressures
dewpoint_A = combined_clean_df['Dewpoint_Grass'].values * units.degC
actual_vp_A = saturation_vapor_pressure(dewpoint_A).to('hPa')

dewpoint_B = combined_clean_df['Dewpoint_Concrete'].values * units.degC
actual_vp_B = saturation_vapor_pressure(dewpoint_B).to('hPa')


## 5 One-Sided Students T-tests

# Data before sunrise
df_no_sun = combined_clean_df[combined_clean_df.index < '2026-10-29 11:55:00']

# Data after sunrise but before cloud cover clears
df_early = combined_clean_df[
    (combined_clean_df.index >= '2026-10-29 11:55:00') & 
    (combined_clean_df.index < '2026-10-29 13:30:00')
]

# Data after cloud cover clears, before Kestrel A was moved
df_mid = combined_clean_df[
    (combined_clean_df.index >= '2026-10-29 13:30:00') & 
    (combined_clean_df.index < '2026-10-29 14:26:00')
]

# Data after Kestrel A was moved
df_late = combined_clean_df[combined_clean_df.index >= '2026-10-29 14:26:00']

# Before Sun Test
if len(df_no_sun) > 1:
    vp_no_sun_A = saturation_vapor_pressure(df_no_sun['Dewpoint_Grass'].values * units.degC).to('hPa').magnitude
    vp_no_sun_B = saturation_vapor_pressure(df_no_sun['Dewpoint_Concrete'].values * units.degC).to('hPa').magnitude
    t_no_sun, p_no_sun = stats.ttest_rel(vp_no_sun_A, vp_no_sun_B, alternative='less')
    print(f"No Sun Period - T-stat: {t_no_sun:.4f}, P-value: {p_no_sun:.4e}")

# Cloudy period test
if len(df_early) > 1:
    vp_early_A = saturation_vapor_pressure(df_early['Dewpoint_Grass'].values * units.degC).to('hPa').magnitude
    vp_early_B = saturation_vapor_pressure(df_early['Dewpoint_Concrete'].values * units.degC).to('hPa').magnitude
    t_early, p_early = stats.ttest_rel(vp_early_A, vp_early_B, alternative='greater')
    print(f"Early Period (Cloudy) - T-stat: {t_early:.4f}, P-value: {p_early:.4e}")

# Post-Cloudy Pre-moving test
if len(df_mid) > 1:
    vp_mid_A = saturation_vapor_pressure(df_mid['Dewpoint_Grass'].values * units.degC).to('hPa').magnitude
    vp_mid_B = saturation_vapor_pressure(df_mid['Dewpoint_Concrete'].values * units.degC).to('hPa').magnitude
    t_mid, p_mid = stats.ttest_rel(vp_mid_A, vp_mid_B, alternative='greater')
    print(f"Mid Period (Clear & Fixed) - T-stat: {t_mid:.4f}, P-value: {p_mid:.4e}")

# Post move
if len(df_late) > 1:
    vp_late_A = saturation_vapor_pressure(df_late['Dewpoint_Grass'].values * units.degC).to('hPa').magnitude
    vp_late_B = saturation_vapor_pressure(df_late['Dewpoint_Concrete'].values * units.degC).to('hPa').magnitude
    t_late, p_late = stats.ttest_rel(vp_late_A, vp_late_B, alternative='greater')
    print(f"Late Period (Post-Move) - T-stat: {t_late:.4f}, P-value: {p_late:.4e}")

# Entire Day
t_stat, p_value = stats.ttest_rel(actual_vp_A.magnitude, actual_vp_B.magnitude, alternative='greater')
print(f"Entire Day - T-stat {t_stat:.4f}, p-value: {p_value:.4e}")

start_time = combined_clean_df.index.min()
end_time = combined_clean_df.index.max()
total_duration = end_time - start_time

# Output to text file for organization
with open(txt_file_output, 'w') as f:
    f.write("EAPS 227 Post-Analysis Results\n\n")

    f.write("Time Frame: \n")
    f.write(f"Dataset Start (UTC): {start_time}\n")
    f.write(f"Dataset End (UTC): {end_time}\n")
    f.write(f"Total Dataset Duration: {total_duration}\n\n")

    f.write("Hypothesis Test Results:\n")
    f.write(f"Overall Paired T-statistic (Vapor Pressure): {t_stat:.4f}\n")
    f.write(f"Overall P-value: {p_value:.4e}\n\n")
    f.write(f"No Sun Period - T-stat: {t_no_sun:.4f}, P-value: {p_no_sun:.4e}\n")
    f.write(f"Early Period (Cloudy) - T-stat: {t_early:.4f}, P-value: {p_early:.4e}\n")
    f.write(f"Mid Period (Clear & Fixed) - T-stat: {t_mid:.4f}, P-value: {p_mid:.4e}\n")
    f.write(f"Late Period (Post-Move) - T-stat: {t_late:.4f}, P-value: {p_late:.4e}\n")

## Graphs

# Plotted Values Section
x = combined_clean_df.index

y1_A = actual_vp_A.magnitude
y1_B = actual_vp_B.magnitude
vp_delta = y1_A - y1_B

y2_A = combined_clean_df['Temp_Grass']
y2_B = combined_clean_df['Temp_Concrete']
temp_delta = y2_A - y2_B

# Vapor Pressure Over Time
plt.figure(figsize=(12, 6))
plt.plot(x, y1_A, label='Vapor Pressure - Grass (A)', color='blue')
plt.plot(x, y1_B, label='Vapor Pressure - Concrete (B)', color='red', alpha=0.7)

plt.xlabel('Time')
plt.ylabel('Vapor Pressure (hPa)')
plt.title('Vapor Pressure Comparison: Grass vs. Concrete')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('vapor_pressure_comparison.png', dpi=300)
plt.close() 

# Temperature Over Time
plt.figure(figsize=(12, 6))
plt.plot(x, y2_A, label='Temperature - Grass (A)', color='orange')
plt.plot(x, y2_B, label='Temperature - Concrete (B)', color='purple', alpha=0.7)

plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Comparison: Grass vs. Concrete')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('temperature_comparison.png', dpi=300)
plt.close()

# Temp, Vapor Pressure, Over Time
plt.figure(figsize=(12, 6))
plt.plot(x, y2_A, label='Temperature - Grass (A)', color='orange')
plt.plot(x, y2_B, label='Temperature - Concrete (B)', color='purple', alpha=0.7)
plt.plot(x, y1_A, label='Vapor Pressure - Grass (A)', color='blue')
plt.plot(x, y1_B, label='Vapor Pressure - Concrete (B)', color='red', alpha=0.7)

plt.xlabel('Time')
plt.ylabel('Temperature (°C), Vapor Pressure (hPa)')
plt.title('Temp and Vapor Pressure Comparison: Grass vs. Concrete')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('temperature_and_vp_comparison.png', dpi=300)
plt.close()

# Vapor Pressure Delta Over Time
plt.figure(figsize=(12, 6))
plt.plot(x, vp_delta, label='VP Difference (Grass - Concrete)', color='green')
plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7) # Reference line at 0

plt.xlabel('Time')
plt.ylabel('Difference in Vapor Pressure (hPa) [Grass - Concrete]')
plt.title('Vapor Pressure Difference Over Time')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('vapor_pressure_delta.png', dpi=300)
plt.close()

# Temperature Delta Over Time
plt.figure(figsize=(12, 6))
plt.plot(x, temp_delta, label='Temp Difference (Grass - Concrete)', color='saddlebrown')
plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7) # Reference line at 0

plt.xlabel('Time')
plt.ylabel('Difference in Temperature (°C) [Grass - Concrete]')
plt.title('Temperature Difference Over Time')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('temperature_delta.png', dpi=300)
plt.close()

# Temp and Vapor Pressure Delta Over Time
plt.figure(figsize=(12, 6))
plt.plot(x, temp_delta, label='Temp Difference (Grass - Concrete)', color='saddlebrown')
plt.plot(x, vp_delta, label='VP Difference (Grass - Concrete)', color='green')
plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7) # Reference line at 0

plt.xlabel('Time')
plt.ylabel('Delta Temperature (°C) and VP (hPa) [Grass - Concrete]')
plt.title('Temperature and Vapor Pressure Differences Over Time')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('temperature_and_vp_delta.png', dpi=300)
plt.close()

# Phase Space Plot (T vs VP)
plt.figure(figsize=(8, 6))
plt.plot(y2_A, y1_A, label='Grass', color='blue', alpha=0.7)
plt.plot(y2_B, y1_B, label='Concrete', color='red', alpha=0.7)
plt.xlabel('Temperature (°C)')
plt.ylabel('Vapor Pressure (hPa)')
plt.title('Phase Space: Temperature vs. Vapor Pressure')
plt.legend()
plt.savefig('phase_space.png', dpi=300)

