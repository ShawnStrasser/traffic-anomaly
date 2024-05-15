# Traffic Anomaly

`traffic_anomaly` is a production ready Python package for robust decomposition and anomaly detection on multiple time series at once. It uses Ibis to integrate with any SQL backend in a production pipeline, or run locally with the included DuckDB backend.

Designed for real world messy traffic data (volumes, travel times), `traffic_anomaly` uses medians to decompose time series into trend, daily, weekly, and residual components. Anomalies are then classified, and Median Absolute Deviation may be used for further robustness. Missing data are handled, and time periods without sufficient data can be thrown out. Check out `example.ipynb` in this repository for a demo.

This package does not produce plots but here's one anyway:

![Example](example_plot.png)

# Installation

Note: Ibis and DuckDB are dependencies and will be installed automatically.

```bash
pip install traffic_anomaly
```

# Considerations

The seasonal components are not allowed to change over time, therefore, it is important to limit the number of weeks included in the model, especially if there is yearly seasonality (and there is). The recommended use for application over a long date range is to run the model incrementally over a rolling window of about 6 weeks.

Because traffic data anomalies usually skew higher, forecasts made by this model are systemically low because in a right tailed distribution the median will be lower than the mean. This is by design, as the model is meant primarily for anomaly detection and not forecasting.

# Notes On Anomaly Detection

`traffic_anomaly` can classify two separate types of anomalies:

1. Entity-Level Anomalies are detected for individual entities based on their own historical patterns, without considering the group context.
2. Group-Level Anomalies are detected for entities when compared to the behavior of other entities within the same group. Group-level anomalies are more rare because in order to be considered for classification as a group-level anomaly, a time period must also have been classified as an entity-level anomaly.

Why is that needed? Well, say you're data is vehicle travel times within a city and there is a snow storm. Travel times across the city drop, and if you're looking at roadway segments in isolation, everything is an anomaly. That's nice, but what if you're only interested in things that are broken? That's where group-level anomalies come in. They are more rare, but they are more likely to be actionable. Probably not much you can do about that snow storm...

# Future Plans/Support
It would be nice to add support for Holidays and a yearly component... please help?

### Change Point Detection
I have working code from the `ruptures` package but it's not integrated here yet, and it's slower than molasses. I'll get to it eventually.