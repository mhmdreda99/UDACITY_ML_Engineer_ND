# Global carbon dioxide emission prediction

Created by: Mohamed Reda 

## Domain background

Global warming is an effect of climate change driven by greenhouse gas emissions increased by human activities. There have been previous periods of climate change, but since the mid-20th century, human activities have had an unprecedented impact on the Earth's climate system causing changes on a global scale [[1]](#1). The biggest global warming factor observed since 1850 is the emission of greenhouse gases, of which more than 90% are carbon dioxide (CO2) and methane, the burning of fossil fuels (coal, oil and gas) for energy consumption is the main source of these emissions, with additional contributions from agriculture, deforestation and manufacturing [[2]](#2).

The increase in temperature on land is about twice the global average increase, leading to the expansion of deserts, heat waves and forest fires [[3]](#3) Increasing rates of evaporation cause more intense storms and extreme weather events [[4]](#4). The increase in temperature is amplified in the Arctic, with the melting of permafrost, glacial retreat and loss of sea ice [[5]](#5). Impacts on ecosystems include the migration or extinction of many species [[6]](#6). Climate change threatens food security and access to water, leads to economic losses and is expected to increase population migration. Risks of floods, infectious diseases and extreme heat are heightened, and the World Health Organization considers climate change to be the greatest threat to global health in the 21st century [[7]](#7), and even if actions to minimize future warming are successful, some effects will continue for centuries, including rising sea levels, rising ocean temperatures and ocean acidification [[8]](#8).

Many of the impacts mentioned are already felt at the current level of warming, which is about 1.1°C (2.0°F) [[9]](#9). The Intergovernmental Panel on Climate Change (IPCC) has published a series of reports that project significant increases in these impacts as warming continues at 1.5°C (2.7°F) [[10]](#10). Responding to climate change involves mitigation and adaptation [[11]](#11). Mitigation - limiting climate change - consists of reducing emissions of greenhouse gases and removing them from the atmosphere; methods include the development and deployment of low-carbon energy sources, such as wind and solar, the elimination of coal and the improvement of energy efficiency, reforestation and forest preservation. Adaptation consists of adjusting to the current or expected climate, such as through improved protection of coastal areas, better disaster management and the development of more resilient crops. Adaptation alone cannot avoid the risk of "serious, widespread and irreversible" impacts [[12]](#12).

## Problem statement

Mitigating carbon dioxide emissions is the challenge of the future to stabilize global warming [[13]](#13). Given this scenario, predictive models can be useful to provide projections of global carbon dioxide emissions, which the results can collaborate with the planning of measures for mitigation and control of future emissions. Therefore, this project aims to implement a predictive model for global carbon dioxide emissions.

## Datasets and inputs

The data set that will be used in this project is available by *Our World in Data* [[14]](#14), The main characteristics of the data set are described below:

* It's a .csv file
* Contains observations since 1750 to 2019
* It has 23.708 rows x 55 columns
* Features `iso_code` and `country` are object type, all others features are numeric

In this project will be used `year` and `co2` features in a univariate time series approach.

## Benchmark model

The scientific article, *Forecasting the carbon dioxide emissions in 53 countries and regions using a non-equigap grey model* [[15]](#15), achieved a MAPE of less than 10%, this will be the baseline for this project.

## Solution statement

The solution proposed for this project is the implementation of DeepAR. The Amazon SageMaker DeepAR forecasting algorithm is a supervised learning algorithm for forecasting scalar (one-dimensional) time series using recurrent neural networks (RNN). 

## Evaluation metrics

The evaluation metric choosed for this project is the mean absolute percentage error (MAPE), is a measure of prediction accuracy of a forecasting method in statistics, for example in trend estimation, also used as a loss function for regression problems in machine learning. It usually expresses the accuracy as a ratio defined by the formula:

![formula](https://latex.codecogs.com/gif.latex?\LARGE&space;MAPE=\frac{1}{n}\sum_{t=1}^{n}&space;\frac{\left&space;|\widehat{y}_{t}-y_{t}&space;\right&space;|}{y_t})

## Project design

The project will be implemented in the following steps:

1) Data Preprocessing
  * `co2` feature will be grouped by `year` for a annualy global forecasting of carbon dioxide emissions
  * Data will be suitable for the DeepAR model
  * 4 sets of data will be generated for training and testing 
2) Training the model
  * Pre-processed data will be sent to s3
  * The estimator object will be defined
  * Hyperparameters will be defined, and the estimator containing DeepAR will be trained with the data
3) Evaluating the model
  * The MAPE of each test set will be calculated, a boxplot will be generated to assess the variation by quartiles and to obtain the median
4) Deploying the model
  * Deploy with the visualization of the global carbon dioxide emission forecast for the next 5 years

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
