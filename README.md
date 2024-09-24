# SenSE-T1DM: Multimodal Sensor Data for Blood Glucose Estimation

![SeNSE Thumbnail](https://github.com/user-attachments/assets/a5bd3e49-118c-4b74-b73b-16ca14fef5f6)

## Overview

The SenSE-T1DM dataset is an open-source collection of multi-modal physiological data designed to improve diabetes management and advance non-invasive glucose monitoring techniques. This dataset addresses the shortcomings of existing resources, which often lack comprehensive data from different physiological signals.

<b>Key features of the SenSE-T1DM dataset:</b>

* It includes data from three types of sensors: Electrocardiography (ECG), Photoplethysmography (PPG), and Continuous Glucose Monitors (CGM).
* The dataset was collected over a period of 17 days from 10 participants diagnosed with Type 1 diabetes.
* The data was gathered in real-world settings, allowing for more practical, everyday-use scenarios.
* It aims to support the development of non-invasive glucose monitoring solutions, reducing the reliance on costly and invasive CGM devices.

For more information, please consult our [paper]().

## Get Started

### Installation
1. Clone the repository
```
git clone https://github.com/Morris88826/SeNSE.git
cd SeNSE
```
2. Install dependent packages
```
conda create --name SeNSE python=3.11
conda activate SeNSE
pip install -r requirements.txt
```

### Download
We are utilizing data from the **[TCH: Cohort 1 Data](https://drive.google.com/drive/folders/1-GshKVAiVbbSJseSHj5Zwiacmss1G0g4?usp=drive_link)** and **[TCH: Cohort 2 Data](https://drive.google.com/drive/folders/1XS1EqnIQl70-pcNLR-fJs4h3rytZcEBb?usp=drive_link)** folders, located on the SeNSE TAMU team drive.

* For access to the ECG data, please contact Professor [Gutierrez-Osuna](mailto:rgutier@cse.tamu.edu) at the PSI Lab, within the Department of Computer Science & Engineering at Texas A&M University.

Each of the **TCH: Cohort 1 Data** and **TCH: Cohort 2 Data** folders should contain five subject subfolders (S01-S05). For each subject, download the following folders:
* zephyr
* e4

For the CGM data, locate the file in the ***cgm*** folder. The file should be named something like ```Clarity_Export_*.csv```. Download this file and rename it to ```cgm.csv```.

Once downloaded, place all the data into the ```./dataset/raw``` directory as follows:
- SeNSE
  - dataset
    - raw
      - c1s01
        - e4
        - zephyr
        - cgm.csv
      - c1s02

### Preprocessing
After downloading all the data and placed in the file structure mentioned above, please run the following command for data preprocessing.
```
python preprocess.py --subject_id <subject_id>
```
This process will combine the multimodal data for each subject, aligning it based on the CGM data segments. The processed data will be saved in the ```./dataset/processed/<subject_id>``` directory, where each file corresponds to a specific CGM segment. 

Please note that the processing time may be significant. Alternatively, you can contact [mtseng@tamu.edu](mailto:rgutier@cse.tamu.edu) to request direct access to the pre-processed data.

## Data
For each processed data, it is saved in a json format that contains multimodal signals within a recorded cgm section (a 5-minute window). In detail, it includes:
* <b>Index</b>
* <b>Timestamp</b>: when the CGM value is being recorded
* <b>glucose</b>: the recorded glucose value
* <b>zephyr</b>:
    * Accel: raw data from Zephyr’s 3-axis accelerometer sensor
        * 'Time', 'Vertical', 'Lateral', 'Sagittal'
    * Breathing: uncalibrated representation of the breathing sensor output (25Hz)
        * 'Time', 'BreathingWaveform'
    * ECG: collected 12-bit filtered ECG signals (250Hz)
        * 'Time', 'EcgWaveform'
    * Summary: other relevant metrics
        * 'Time', 'HR', 'BR', 'Posture', 'Activity', 'HRConfidence', 'ECGNoise'
* <b>e4</b>: 
    * ACC: data from e4's 3-axis accelerometer sensor.
        * 'Time', 'x', 'y', 'z'
    * HR: average heart rate extracted from the BVP signal
        * 'Time', 'HR'
    * BVP: data from photoplethysmography (PPG)
        * 'Time', 'BVP'
    * EDA: data from the electrodermal activity sensor
        * 'Time', 'EDA'
    * TEMP: data from temperature sensor (°C)
        * 'Time', 'TEMP'

We have also included a [jupyter notebook](./notebooks/visualize.ipynb) that provides an interactive demo for visualizing the signal. This notebook walks you through key steps in analyzing and processing the data, allowing you to explore and better understand the signal.

## Citation
Details to be added soon.


## Help
If you have any questions, please contact [mtseng@tamu.edu](mailto:rgutier@cse.tamu.edu).
