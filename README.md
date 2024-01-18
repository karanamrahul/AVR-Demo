# Altumint AVR Demo

Altumint AVR (Automatic Vehicle Recognition) Demo is a state-of-the-art real-time vehicle and person recognition system designed to enhance monitoring and security operations across various industries.

## Features

### Real-Time Recognition and Database Integration
- **Dynamic License Plate Tracking**: Real-time identification of license plates in multiple traffic lanes and at high speeds.
- **Intelligent Archiving**: Automated storage of detected plates with immediate database query capabilities.

### Advanced Recognition and Surveillance Analytics
- **Facial Recognition Technology**: Identifies persons of interest and flags unusual activities.
- **Stolen Vehicle Detection**: Analytical tools to detect and alert on stolen vehicles.

### Interactive Mapping and Intelligent Search Capabilities
- **Surveillance Device Mapping**: Live visualization of surveillance devices for real-time vehicle tracking.
- **Text-to-Video Analytics**: Advanced search analysis to pinpoint vehicle locations using textual descriptions.


## Install

Install required packages with pip:

```shell
pip install -r requirements.txt
```

or with conda:

```shell
conda env create -f environment.yml

# activate the conda environment
conda activate altumint_demo
```

## Download weights

Download the model weights：

[det_weights](https://drive.google.com/drive/folders/1OmxDFzY65rj5Nxtdz4S1XWT2QdrkwKZr?usp=share_link)
[pose](https://drive.google.com/drive/folders/1XjlLCDhuuDNfYXmPo2rMr_RUSUDp2WMI?usp=share_link)
[segmentation](https://drive.google.com/drive/folders/1tLPndFlSsV9SR8JKG2I2ijLuqsWkcHzt?usp=share_link)

The model files are saved in the **weights/** folder.

## Run

```shell
python main.py
```

