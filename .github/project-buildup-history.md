# Project Buildup History: Property Valuation Model

- Repository: `property-valuation-model`
- Category: `data_science`
- Subtype: `prediction`
- Source: `project_buildup_2021_2025_daily_plan_extra.csv`
## 2022-09-26 - Day 2: Feature engineering

- Task summary: Started working on the Property Valuation Model feature engineering today. The raw dataset had latitude and longitude but no derived location features. Computed distance to city center, distance to nearest school, and distance to nearest hospital using the Haversine formula. Also extracted the age of the property from the build year and computed a renovation recency indicator from the last renovation year. These spatial and temporal features added meaningfully to the feature set beyond the basic property characteristics.
- Deliverable: Spatial distance features and temporal property features added. Feature count went from 9 to 16.
## 2022-09-26 - Day 2: Feature engineering

- Task summary: Found that city center coordinates had been hardcoded to the wrong location — off by about 3km. Fixed after cross-checking with a map reference.
- Deliverable: City center reference point corrected. Distance features recalculated.
## 2022-10-03 - Day 3: Baseline model

- Task summary: Trained the first baseline model for the Property Valuation project today. Used a simple linear regression first to understand the baseline relationship, then compared with a gradient boosting model. The boosting model's advantage was clear on the validation set — capturing the non-linear interaction between location features and square footage that the linear model could not. Residual analysis showed the model still struggled with premium waterfront properties. Added that as a known limitation note.
- Deliverable: Baseline comparison done. Gradient boosting selected. Waterfront limitation noted.
## 2022-11-07 - Day 4: Model evaluation

- Task summary: Comprehensive model evaluation session for Property Valuation today. Ran evaluation across multiple cuts — overall RMSE and MAE, error by property type, error by price band, and geographic error heatmap. The geographic heatmap revealed that errors were higher in one suburb that had undergone significant recent development — the model had no signal for new construction premium. Added a flag for post-2020 build year as a proxy and retested.
- Deliverable: Evaluation by multiple cuts done. New construction flag added. Error in one suburb reduced.
