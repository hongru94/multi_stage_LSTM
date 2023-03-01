# Incorporating variant frequencies data into short-term forecasting for COVID-19 cases and deaths in the USA: a deep learning approach
This repository provides the code and data for the Multi-stage LSTM model to Forecast Short-Term COVID-19 Cases and Deaths in the US!
### The paper can be found here: (https://www.medrxiv.org/content/10.1101/2022.08.23.22279132v1).

![Network architecture ![image](https://user-images.githubusercontent.com/47940478/202015773-ece717e7-743d-4e07-b3c8-00543abf4682.png)
](https://github.com/hongru94/multi_stage_LSTM/blob/main/figures/figure_1.png?raw=true "Multi-stage LSTM model")

A) Network architecture of the multi-stage LSTM model. B) Prediction structure of the multi-stage LSTM model. At the initial stage, the model uses the most recent data as input, then at the later stage, the model adapts previous prediction as input to make further predictions. The transparent colors represent the model’s output, and solid colors represents the model’s inputs. C) An example forecasting of the multi-stage LSTM model.

### Data
| State-Level Data| Data Preprocessing | Data Source |
| :---:         |     :---:      |          :--- |
| COVID-19 cases/deaths| Raw     | https://github.com/CSSEGISandData/COVID-19  |
| Growth rate of cases/deaths| Raw       | https://github.com/CSSEGISandData/COVID-19|
| Vaccination coverage | Raw  |  https://www.cdc.gov/nhsn/covid19/dial-vaccination-dashboard.html|
| Hospitalization data | Raw | https://www.hhs.gov/index.html |
|Importation risk| Derived| https://www.safegraph.com/, https://github.com/CSSEGISandData/COVID-19 |
|Mobility ratio| Derived | https://www.safegraph.com/ |
|Visits ratio for 21 different destinations| Derived | https://www.safegraph.com/ | 
|COVID-like symptoms in community| Raw | https://github.com/cmu-delphi/delphi-epidata |
| Temperature & Percipitation | Raw | https://github.com/CSSEGISandData/COVID-19_Unified-Dataset |
| Demographic data| Raw | https://www.census.gov/data.html | 
| Variant cases | Derived | https://gisaid.org/ |

Please refer to the Appendix for detailed preprocessing for derived metrics. 


### Model
- Main model: Predict the target epidemiological variables of interest!
- Feature model: Predict independent features to populate the data streams used as input in the main model!
- Prediction: The multi-stage model builds off the initial first stage prediction to forecast an additional week out and continues to implement this iterative approach one stage at a time, to predict further into the future.

### Contributors
Hongru Du, Ensheng Dong, Hamada S. Badr, Mary E. Petrone, Nathan D. Grubaugh, and Lauren M. Gardner

### BibTeX
```latex
@article{du2023incorporating,
  title={Incorporating variant frequencies data into short-term forecasting for COVID-19 cases and deaths in the USA: a deep learning approach},
  author={Du, Hongru and Dong, Ensheng and Badr, Hamada S and Petrone, Mary E and Grubaugh, Nathan D and Gardner, Lauren M},
  journal={EBioMedicine},
  volume={89},
  year={2023},
  publisher={Elsevier}
}
```
