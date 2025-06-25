## Homework

The goal of this homework is to familiarize users with monitoring for ML batch services, using PostgreSQL database to store metrics and Grafana to visualize them.



## Q1. Prepare the dataset

Start with `baseline_model_nyc_taxi_data.ipynb`. Download the March 2024 Green Taxi data. We will use this data to simulate a production usage of a taxi trip duration prediction service.

What is the shape of the downloaded data? How many rows are there?

* 72044
* 78537 
* 57457
* 54396

### Q1. Solution

**57457**

### Q2. Metric

Let's expand the number of data quality metrics we’d like to monitor! Please add one metric of your choice and a quantile value for the `"fare_amount"` column (`quantile=0.5`).

Hint: explore evidently metric `ColumnQuantileMetric` (from `evidently.metrics import ColumnQuantileMetric`) 

What metric did you choose?

### Q2. Solution

To expand the data quality monitoring, I added mainly this metric:

**ColumnSummaryMetric**

This provides a high-level overview of the "fare_amount" column, including statistics like mean, standard deviation, min/max values, and the number of missing values. It's helpful for spotting anomalies or shifts in the distribution at a glance.

`ColumnQuantileMetric` with quantile=0.5 – This captures the median of the "fare_amount" column, which is a robust central tendency measure and helps identify distribution changes that may not affect the mean but shift the data overall.

Chosen metric: ColumnSummaryMetric
Quantile monitored: 0.5 (median)

These two metrics together offer both a general summary and a focused statistical insight into the "fare_amount" column.

## Q3. Monitoring

Let’s start monitoring. Run expanded monitoring for a new batch of data (March 2024). 

What is the maximum value of metric `quantile = 0.5` on the `"fare_amount"` column during March 2024 (calculated daily)?

* 10
* 12.5
* 14.2
* 14.8

### Q3. Solution

**14.2**

## Q4. Dashboard

Finally, let’s add panels with new added metrics to the dashboard. After we customize the  dashboard let's save a dashboard config, so that we can access it later. Hint: click on “Save dashboard” to access JSON configuration of the dashboard. This configuration should be saved locally.

Where to place a dashboard config file?

* `project_folder` (05-monitoring)
* `project_folder/config`  (05-monitoring/config)
* `project_folder/dashboards`  (05-monitoring/dashboards)
* `project_folder/data`  (05-monitoring/data)


### Q4. Solution

I've added a dashboard with a single panel. Although there isn't an explicit "Save dashboard" button in the interface, I referred to the folder structure used in the lessons repository for guidance.

It appears that Grafana commonly uses a dashboards/ folder for storing dashboard configurations. Additionally, the folder structure provided in the lessons also includes a dashboards/ directory.

Based on this, I decided to place the dashboard configuration file in:

`project_folder/dashboards` **(05-monitoring/dashboards)**

This aligns with both standard Grafana usage and the lesson's project structure.