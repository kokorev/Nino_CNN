# Convolutional Neural Network for long-range Nino3.4 index forecasting. 

This model presented here is based on the original idea by Yoo-Geun Ham, Jeong-Hwan Kim, and Jing-Jia Luo [DOI:10.1038/s41586-019-1559-7](https://doi.org/10.1038/s41586-019-1559-7)

## Running the model

Model and prepared input data located in **nino_cnn** folder. It is known to work with python 3.6 and tensorflow 2. Also depends on netCDF4, pandas and numpy for reading the input. 

**/data** - contain input data
	**model.py** - contains function that return keras model.
	**reader.py** - contains class that feeds input data into the model
	**train.py** - runs the model
	

## Data Preparation

Ocean surface temperature and heat content data is taken from Ocean Reanalysis System 5 (ORAS5) by [ECMWF](https://www.ecmwf.int/en/research/climate-reanalysis/ocean-reanalysis). Monthly fields of two variables is used - Sea Surface Temperature (SST) and Total Heat content in upper 300 meters (HC300). For period from 1979 to 2018 ORAS5 provides reanalysis result for five ensemble members, plus backward extension data available for 1950-1978 for a single ensemble member. Data from all of the ensemble members is used to train model.

The original data is regrided to Gaussian N16 grid (32x64) approximately 5x5 degrees. Two southmost data rows that cover Antarctica are thrown away since they have no data, resulting in 30x64 final grid. The data was normalized by applying Max-Min Feature Scaling. Every cell was processed independently, meaning that 0 and 1 represent minimum and maximum in particular scale rather than global once. 

Scripts for downloading and processing the raw data provided in **prepare_data** directry. Resulting input data can be found in **nino_cnn/data**

## Training

The following training procedure is used. The model initially pre-trained using ORAS5 backward extension data from 1950-1979, for 10 epochs. Backward extension data is considerably lower quality and only used to provide a good starting point for a training and make training results a more stable. Next, for each ORAS5 ensemble member model is trained data for 1979-2008 for 15 epochs. The data from 2009-2018 period is left for verification and data from 2019 to present reserved for a final assessment.

## Results
Nino3.4 prediction (orange) and true values (blue)

6-month lead time,  correlation - 0.88:
![alt text](https://github.com/kokorev/Nino_CNN/raw/master/img/prediction_6m_corr088.png "6-month lead time prediction")

8-month lead time,  correlation - 0.80:
![alt text](https://github.com/kokorev/Nino_CNN/raw/master/img/prediction_8m_corr080.png "8-month lead time prediction")

12-month lead time,  correlation - 0.60:
![alt text](https://github.com/kokorev/Nino_CNN/raw/master/img/prediction_12m_corr060.png "12-month lead time prediction")