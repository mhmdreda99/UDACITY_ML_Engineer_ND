# Global carbon dioxide emission prediction

Project proposal by Mohamed Reda

## Domain background

Global warming is an effect of climate change driven by greenhouse gas emissions increased by human activities. There have been previous periods of climate change, but since the mid-20th century, human activities have had an unprecedented impact on the Earth's climate system causing changes on a global scale [[1]](#1). The biggest global warming factor observed since 1850 is the emission of greenhouse gases, of which more than 90% are carbon dioxide (CO2) and methane, the burning of fossil fuels (coal, oil and gas) for energy consumption is the main source of these emissions, with additional contributions from agriculture, deforestation and manufacturing [[2]](#2).

The increase in temperature on land is about twice the global average increase, leading to the expansion of deserts, heat waves and forest fires [[3]](#3) Increasing rates of evaporation cause more intense storms and extreme weather events [[4]](#4). The increase in temperature is amplified in the Arctic, with the melting of permafrost, glacial retreat and loss of sea ice [[5]](#5). Impacts on ecosystems include the migration or extinction of many species [[6]](#6). Climate change threatens food security and access to water, leads to economic losses and is expected to increase population migration. Risks of floods, infectious diseases and extreme heat are heightened, and the World Health Organization considers climate change to be the greatest threat to global health in the 21st century [[7]](#7), and even if actions to minimize future warming are successful, some effects will continue for centuries, including rising sea levels, rising ocean temperatures and ocean acidification [[8]](#8).

Many of the impacts mentioned are already felt at the current level of warming, which is about 1.1°C (2.0°F) [[9]](#9). The Intergovernmental Panel on Climate Change (IPCC) has published a series of reports that project significant increases in these impacts as warming continues at 1.5°C (2.7°F) [[10]](#10). Responding to climate change involves mitigation and adaptation [[11]](#11). Mitigation - limiting climate change - consists of reducing emissions of greenhouse gases and removing them from the atmosphere; methods include the development and deployment of low-carbon energy sources, such as wind and solar, the elimination of coal and the improvement of energy efficiency, reforestation and forest preservation. Adaptation consists of adjusting to the current or expected climate, such as through improved protection of coastal areas, better disaster management and the development of more resilient crops. Adaptation alone cannot avoid the risk of "serious, widespread and irreversible" impacts [[12]](#12).

## Problem statement

Mitigating carbon dioxide emissions is the challenge of the future to stabilize global warming [[13]](#13). Given this scenario, predictive models can be useful to provide projections of global carbon dioxide emissions, which the results can collaborate with the planning of measures for mitigation and control of future emissions. Therefore, this project aims to implement a predictive model for global carbon dioxide emissions.

## Evaluation metrics

The evaluation metric choosed for this project is the mean absolute percentage error (MAPE), is a measure of prediction accuracy of a forecasting method in statistics, for example in trend estimation, also used as a loss function for regression problems in machine learning. It usually expresses the accuracy as a ratio defined by the formula:

![formula](https://latex.codecogs.com/gif.latex?\LARGE&space;MAPE=\frac{1}{n}\sum_{t=1}^{n}&space;\frac{\left&space;|\widehat{y}_{t}-y_{t}&space;\right&space;|}{y_t})

## Data exploration

The data set that will be used in this project is available by *Our World in Data* [[14]](#14), The main characteristics of the data set are described below:

* It's a .csv file
* Contains observations since 1750 to 2019
* It has 23.708 rows x 55 columns
* Features `iso_code` and `country` are object type, all others features are numeric

In this project will be used `year` and `co2` features in a univariate time series approach.

## Exploratory Visualization

The time series of the global carbon dioxide emission was decomposed to assess its seasonality, trend and error.

![alt text](https://github.com/plbalmeida/udacity-ml-engineer/blob/main/capstone/img/ts_observed.png)
![alt text](https://github.com/plbalmeida/udacity-ml-engineer/blob/main/capstone/img/ts_seasonal.png)
![alt text](https://github.com/plbalmeida/udacity-ml-engineer/blob/main/capstone/img/ts_trend.png)
![alt text](https://github.com/plbalmeida/udacity-ml-engineer/blob/main/capstone/img/ts_residuals.png)

Running the plots of the observed, trend, seasonal, and residual time series, we can see that the results doesn't interesting. So let's try another approach to extract some insights about the time series. Next, the amount of carbon dioxide emission will be analyzed every 50 years, since 1850.

<p align="center">
  <img src="https://github.com/plbalmeida/udacity-ml-engineer/blob/main/capstone/img/emissions_by_period.png">
</p>

In the boxplot it is possible to observe the variation in the quartiles of carbon dioxide emission in the periods from 1850 to 1900, 1900 to 1950, 1950 to 2000, and 2000 to 2019. The annual averages of global carbon dioxide emissions in the period from 1950 to 2000 increased by 1,688% compared to the period from 1850 to 1900, and in the period from 2000 to 2019 emissions increased by 3,176%.

Another interesting approach could be check the percentual variation year by year.

![alt text](https://github.com/plbalmeida/udacity-ml-engineer/blob/main/capstone/img/emissions_percentual_change.png)

We can conclude so far that, on average, the percentage change from year to year is approximately 3%.

## Algorithms and Techniques

The Amazon SageMaker DeepAR forecasting algorithm is a supervised learning algorithm for forecasting scalar (one-dimensional) time series using recurrent neural networks (RNN). In the following figure, a recurrent neural network, with a hidden state that is meant to carry pertinent information from one input item in the series to others.

<p align="center">
  <img src="https://miro.medium.com/max/2640/1*aIT6tmnk3qHpStkOX3gGcQ.png">
</p>

DeepAR is a methodology for producing accurate probabilistic forecasts, based on training an auto-regressive recurrent network model on a large number of related time series. According to the authors who developed the DeepAR model, it was possible to demonstrate how by applying deep learning techniques to forecasting, one can overcome many of the challenges faced by widely-used classical approaches to the problem. The scientific article shows how through extensive empirical evaluation on several real-world forecasting data sets accuracy improvements of around 15%
compared to state-of-the-art methods [[16]](#16).

During training, DeepAR accepts a training dataset and an optional test dataset. It uses the test dataset to evaluate the trained model. In general, the datasets don't have to contain the same set of time series. It's possible use a model trained on a given training set to generate forecasts for the future of the time series in the training set, and for other time series. Both the training and the test datasets consist of one or, preferably, more target time series. Each target time series can optionally be associated with a vector of feature time series and a vector of categorical features.

For example, the following is an element of a training set indexed by i which consists of a target time series, Zi,t, and two associated feature time series, Xi,1,t and Xi,2,t}:

<p align="center">
  <img src="https://docs.aws.amazon.com/sagemaker/latest/dg/images/ts-full-159.base.png">
</p>

The target time series might contain missing values, which are represented by line breaks in the time series. DeepAR supports only feature time series that are known in the future. This allows you to run "what if?" scenarios. What happens, for example, if I change the price of a product in some way?

Each target time series can also be associated with a number of categorical features. You can use these features to encode which groupings a time series belongs to. Categorical features allow the model to learn typical behavior for groups, which it can use to increase model accuracy. DeepAR implements this by learning an embedding vector for each group that captures the common properties of all time series in the group.

## Benchmark

The scientific article, *Forecasting the carbon dioxide emissions in 53 countries and regions using a non-equigap grey model* [[15]](#15), achieved a MAPE of less than 10%, this will be the baseline for this project.

## Solution statement

The solution proposed for this project is the implementation of DeepAR. The Amazon SageMaker DeepAR forecasting algorithm is a supervised learning algorithm for forecasting scalar (one-dimensional) time series using recurrent neural networks (RNN) to predict global carbon dioxide emissions.

## Data Preprocessing

`co2` feature was grouped by `year` for a annualy global forecasting of carbon dioxide emissions.

The data were suitable for the DeepAR model, following the pattern in joson format:

`
{"start": ..., "target": [1, 5, 10, 2], "dynamic_feat": [[0, 1, 1, 0]]}
`

4 sets of data was generated for training and testing. 

## Implementation

Pre-processed data was sent to S3 bucket.

The estimator object was defined.

```python
estimator = sagemaker.estimator.Estimator(
    image_uri=image_name,
    sagemaker_session=sagemaker_session,
    role=role,
    train_instance_count=1,
    train_instance_type='ml.c4.2xlarge',
    base_job_name='deepar-co2-emission',
    output_path=s3_output_path
)
```
  
## Refinement

Hyperparameters was defined, and the estimator containing DeepAR was trained with the data.

```python
freq = '12M'
hyperparameters = {
    'time_freq': freq,
    'epochs': '400',
    'early_stopping_patience': '40',
    'mini_batch_size': '64',
    'learning_rate': '5E-4',
    'context_length': '5',
    'prediction_length': '5'
}
```

## Model Evaluation and Validation

The method that can be used for cross-validating the time-series model is cross-validation on a rolling basis, or *Time Series Split Cross-Validation*. Start with a small subset of data for training purpose, forecast for the later data points and then checking the accuracy for the forecasted data points. The same forecasted data points are then included as part of the next training dataset and subsequent data points are forecasted. What we need to do is to create 4 pairs of training/test sets that follow those two rules:

* every test set contains unique observations
* observations from the training set occur before their corresponding test set 

Source: https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4.

<p align="center">
  <img src="https://github.com/plbalmeida/udacity-ml-engineer/blob/main/capstone/img/time-series-cross-validation.png">
</p>

The evaluation metric choosed for this project is the mean absolute percentage error (MAPE), model errors were calculated using this metric for the four sets of test data.

<p align="center">
  <img src="https://github.com/plbalmeida/udacity-ml-engineer/blob/main/capstone/img/test_0.png">
</p>

<p align="center">
  <img src="https://github.com/plbalmeida/udacity-ml-engineer/blob/main/capstone/img/test_1.png">
</p>

<p align="center">
  <img src="https://github.com/plbalmeida/udacity-ml-engineer/blob/main/capstone/img/test_2.png">
</p>

<p align="center">
  <img src="https://github.com/plbalmeida/udacity-ml-engineer/blob/main/capstone/img/test_3.png">
</p>

<p align="center">
  <img src="https://github.com/plbalmeida/udacity-ml-engineer/blob/main/capstone/img/mape.png">
</p>

## Justification

MAPE for test is around 2.29%, it's under the benchmark found of 10%. Follow the link with the deployment of predictions for the following years: http://capstone-deploy.s3-website-sa-east-1.amazonaws.com/

## References

<a id="1">[1]</a> Knutson, T.; Kossin, J. P.; Mears, C.; Perlwitz, J.; Wehner, M. F. (2017). In https://science2017.globalchange.gov/downloads/CSSR_Ch3_Detection_and_Attribution.pdf. 

<a id="2">[2]</a> US EPA, 21 February 2021. In https://www.epa.gov/ghgemissions/overview-greenhouse-gases.

<a id="3">[3]</a> IPCC (2019). Shukla, P. R.; Skea, J.; Calvo Buendia, E.; Masson-Delmotte, V.; et al. (eds.). IPCC Special Report on Climate Change, Desertification, Land Degradation, Sustainable Land Management, Food Security, and Greenhouse gas fluxes in Terrestrial Ecosystems. In https://www.ipcc.ch/site/assets/uploads/2019/11/SRCCL-Full-Report-Compiled-191128.pdf.

<a id="4">[4]</a> Kossin, J. P.; Hall, T.; Knutson, T.; Kunkel, K. E.; Trapp, R. J.; Waliser, D. E.; Wehner, M. F. (2017). "Chapter 9: Extreme Storms". In https://science2017.globalchange.gov/chapter/9/. 

<a id="5">[5]</a> IPCC (2019). Pörtner, H.-O.; Roberts, D. C.; Masson-Delmotte, V.; Zhai, P.; et al. (eds.). In https://www.ipcc.ch/site/assets/uploads/sites/3/2019/12/SROCC_FullReport_FINAL.pdf.

<a id="6">[6]</a> US EPA, 19 January 2017. In https://19january2017snapshot.epa.gov/climate-impacts/climate-impacts-ecosystems_.html#Extinction.

<a id="7">[7]</a> IPCC AR5 SYR (2014). The Core Writing Team; Pachauri, R. K.; Meyer, L. A. (eds.). Climate Change 2014: Synthesis Report. Contribution of Working Groups I, II and III to the Fifth Assessment Report of the Intergovernmental Panel on Climate Change. Geneva, Switzerland: IPCC. In https://www.ipcc.ch/report/ar5/syr/.

<a id="8">[8]</a> Allen, M. R.; Dube, O. P.; Solecki, W.; Aragón-Durand, F.; et al. (2018). In https://www.ipcc.ch/site/assets/uploads/sites/2/2019/05/SR15_Chapter1_High_Res.pdf. 

<a id="9">[9]</a> Lindsey, R.; Dahlman, L. (14 August 2020). In https://www.climate.gov/news-features/understanding-climate/climate-change-global-temperature.

<a id="10">[10]</a> IPCC (2018). In https://www.ipcc.ch/site/assets/uploads/sites/2/2019/05/SR15_SPM_version_report_HR.pdf.

<a id="11">[11]</a> NASA, Mitigation and Adaptation 2020. In https://web.archive.org/web/20210104032553/https://climate.nasa.gov/solutions/adaptation-mitigation/.

<a id="12">[12]</a> IPCC AR5 SYR (2014). The Core Writing Team; Pachauri, R. K.; Meyer, L. A. (eds.). Climate Change 2014: Synthesis Report. Contribution of Working Groups I, II and III to the Fifth Assessment Report of the Intergovernmental Panel on Climate Change. Geneva, Switzerland: IPCC.

<a id="13">[13]</a> Kainuma M.; Matsuoka, Y.; Morita, T. (2000) The AIM/end-use model and its application to forecast Japanese carbon dioxide emissions Eur. J. Oper. Res. 122 416–25 

<a id="14">[14]</a> Data on CO2 and Greenhouse Gas Emissions, 21 February 2021. In https://github.com/owid/co2-data.

<a id="15">[15]</a> Xu, Z., Liu, L. & Wu, L. Forecasting the carbon dioxide emissions in 53 countries and regions using a non-equigap grey model. Environ Sci Pollut Res (2020). https://doi.org/10.1007/s11356-020-11638-7. In https://link.springer.com/article/10.1007/s11356-020-11638-7#Tab3.

<a id="16">[16]</a> Valentin Flunkert, David Salinas, and Jan Gasthaus. DeepAR: Probabilistic forecasting with autoregressive recurrent networks. CoRR, abs/1704.04110, 2017. URL http://arxiv.org/abs/1704.04110.
