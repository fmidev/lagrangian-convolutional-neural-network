# callbacks
Callbacks are modules passed to the Pytorch Lightning Trainer object, containing non-essential or additional functionality. 

## Implemented callbacks 

- `log_nowcast_callback.py`
    - Plot to the logger (for ex. Tensorboard) radar images for observations and predictions. 
- `nowcast_metrics_callback.py`
    - Calculate Nowcasting skill metrics for testing and validation, plot them as a function of leadtime and potentially save them to pictures/csv files.