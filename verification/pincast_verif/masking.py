"""
Bent Harnist 12.04.2022
"""

import numpy as np

    
def _make_mask(predictions: dict, n_leadtimes: int) -> np.ndarray:
    pred_names = list(predictions.keys())
    
    mask = np.zeros((n_leadtimes, *predictions[pred_names[0]].shape[-2:]), dtype=bool)
    
    for pred in pred_names:
        if mask.ndim == predictions[pred].ndim:
            mask = np.logical_or(np.isnan(predictions[pred]), mask)
        elif mask.ndim == predictions[pred].ndim - 1:
            for ens_member in range(predictions[pred].shape[1]):
                mask = np.logical_or(np.isnan(predictions[pred][:,ens_member]), mask)
        else:
            raise ValueError(f"Bad pred array number of dims : {predictions[pred].ndim}")

    return mask
    


def _apply_mask(predictions: dict, mask: np.ndarray) -> dict:
    masked_predictions = predictions.copy()
    for pred in predictions.keys():
        preds = np.stack(masked_predictions[pred], axis=0)
        if preds.shape == mask.shape:
            preds[mask] = np.nan
        else:
            for ens_member in range(preds.shape[1]):
                ens_mem = preds[:,ens_member]
                ens_mem[mask] = np.nan
                preds[:,ens_member] = ens_mem
        masked_predictions[pred] = preds
    return masked_predictions


def mask(predictions: dict, n_leadtimes: int) -> dict:
    """Take a dictionary of predictions, and return these predictions
    masked with the union of nan values of all predictions at a
    specific leadtime

    Args:
        predictions (dict): "method" : lt*x*y np.array preds in

    Returns:
        dict: "method" : lt*x*y np.array masked preds out
    """
    mask = _make_mask(predictions=predictions, n_leadtimes=n_leadtimes)
    masked_prediction = _apply_mask(predictions=predictions, mask=mask)
    return masked_prediction
