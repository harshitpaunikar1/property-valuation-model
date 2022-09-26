# Project Buildup History: Property Valuation Model

- Repository: `property-valuation-model`
- Category: `data_science`
- Subtype: `prediction`
- Source: `project_buildup_2021_2025_daily_plan_extra.csv`
## 2022-09-26 - Day 2: Feature engineering

- Task summary: Started working on the Property Valuation Model feature engineering today. The raw dataset had latitude and longitude but no derived location features. Computed distance to city center, distance to nearest school, and distance to nearest hospital using the Haversine formula. Also extracted the age of the property from the build year and computed a renovation recency indicator from the last renovation year. These spatial and temporal features added meaningfully to the feature set beyond the basic property characteristics.
- Deliverable: Spatial distance features and temporal property features added. Feature count went from 9 to 16.
