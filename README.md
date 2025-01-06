# PhysioCGM: A Multimodal Physiological Dataset for Non-Invasive Blood Glucose Estimation


![SeNSE Thumbnail](https://github.com/user-attachments/assets/03450364-244d-462e-8907-fb2ae5f7c765)

## Overview

The PhysioCGM dataset is an open-source collection of multi-modal physiological data designed to improve diabetes management and advance non-invasive glucose monitoring techniques. This dataset addresses the shortcomings of existing resources, which often lack comprehensive data from different physiological signals.

<b>Key features of the PhysioCGM dataset:</b>

* It includes data from three types of sensors: Electrocardiography (ECG), Photoplethysmography (PPG), and Continuous Glucose Monitors (CGM).
* The dataset was collected over 17 days from 10 participants diagnosed with Type 1 diabetes.
* The data was gathered in real-world settings, allowing for more practical, everyday-use scenarios.
* It aims to support the development of non-invasive glucose monitoring solutions, reducing the reliance on costly and invasive CGM devices.

For more information, please consult our [paper]().

## Get Started

### Installation
1. Clone the repository
```
git clone https://github.com/PSI-TAMU/PhysioCGM.git
cd PhysioCGM
```
2. Install dependent packages
```
conda create --name SeNSE python=3.11
conda activate SeNSE
pip install -r requirements.txt
```

### Download
The dataset can be downloaded from the TAMU PSI Lab team drive ([here](https://drive.google.com/drive/folders/1mhMKXQ0gxlSJbl-QSILOjKy6GzHqUGJc?usp=drive_link))

* For access to the data, please contact Professor [Gutierrez-Osuna](mailto:rgutier@cse.tamu.edu) at the PSI Lab, within the Department of Computer Science & Engineering at Texas A&M University.

Once downloaded, place all the data under the ```./dataset``` directory as follows:
- PhysioCGM
  - dataset
    - raw
      - c1s01
        - e4
        - zephyr
        - cgm.csv
      - c1s02
    - processed
      - c1s01
        - 0.pkl
        - 1.pkl
        - 2.pkl
        - ...
      - c1s02
        

### Preprocessing
After downloading the raw data and placed in the file structure mentioned above, one can run the following command for data preprocessing (if not download the processed folder).
```
python preprocess.py --subject_id <subject_id>
```
This process will combine the multimodal data for each subject, aligning it based on the CGM data segments. The processed data will be saved in the ```./dataset/processed/<subject_id>``` directory, where each file corresponds to a specific CGM segment. 

Please note that the processing time may be significant.

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
