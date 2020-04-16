# Instructions to run code on Google Colab

*See the end of this file for a snippet you can copy + paste into a cell in Colab and run.
The next parts of this file explain what happens in that snippet.*

## Install Kaggle API
At the time of writing this, the Kaggle API version on Colab is `1.5.4`, but the latest version (`1.5.6`) is required to download the dataset.
We need to uninstall the current version, and re-install the latest version using `pip`:

```
!pip uninstall kaggle
!pip install kaggle
!kaggle --version
```

Check the output to ensure version `1.5.6` (or later) is installed.

## Add Kaggle API Credentials
You will need to provide API credentials to be able to download the dataset using the Kaggle API.
Get your API credentials as describe in the [Kaggle API docs](https://github.com/Kaggle/kaggle-api#api-credentials) (you should end up with a `kaggle.json` file).

Upload the `kaggle.json` file you get to Colab. The following lines will move the file to the location that the Kaggle API expects it to be in:

```
!mkdir -p /root/.kaggle
!mv kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
```

## Download and extract dataset

Data from the `airbus-ship-detection` competition is downloaded, and extracted into the folder `airbus-data`.
The downloaded zip file is then deleted to save space.

```
!kaggle competitions download airbus-ship-detection
!mkdir -p airbus-data
!unzip airbus-ship-detection.zip -d airbus-data -q
!rm airbus-ship-detection.zip
```


## Copy-and-Paste-able Snippet
```
# Reinstall latest version of Kaggle API
!pip uninstall kaggle
!pip install kaggle

# Move credentials file
!mkdir -p /root/.kaggle
!mv kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json

# Download and extract data
!kaggle competitions download airbus-ship-detection
!mkdir -p airbus-data
!unzip airbus-ship-detection.zip -d airbus-data -q
!rm airbus-ship-detection.zip
```
