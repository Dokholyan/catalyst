import numpy as np

from typing import Union, Callable, Dict, List, Any
import scipy.stats as st


def get_bootstraped_metric(bootstraped_targets: List[List[Any]],
                           bootstraped_predictions, metric):
    bootstraped_metric = []
    for targets, predictions in zip(bootstraped_targets, bootstraped_predictions):
        bootstraped_metric.append(metric(targets, predictions))
    bootstraped_metric = np.array(bootstraped_metric)
    return bootstraped_metric


def get_confidence_intervals(bootstraped_metric: List[float],
                             estimate: float,
                             alpha: int = 0.05,
                             mode: str = 'central'):
    assert mode in ['central', 'normal', 'percentiles']
    if mode == 'normal':
        z_alpha = st.norm.ppf(1 - alpha / 2, loc=0.0, scale=1.0)
        se = np.std(bootstraped_metric)
        left = estimate - z_alpha * se
        right = estimate + z_alpha * se
    bootstraped_metric = np.sort(bootstraped_metric)
    N_bootstrap = len(bootstraped_metric)
    if mode == 'percentiles':
        left = bootstraped_metric[int(alpha / 2 * N_bootstrap)]
        right = bootstraped_metric[int((1 - alpha / 2) * N_bootstrap)]
    if mode == 'central':
        left = 2 * estimate - bootstraped_metric[int((1 - alpha / 2) * N_bootstrap)]
        right = 2 * estimate - bootstraped_metric[int(alpha / 2 * N_bootstrap)]
    return (left, right)


def bootstrap_evaluation(targets: List[Any],
                         predictions: List[Any],
                         metrics: Union[Callable, List[Callable], Dict[str, Callable]],
                         alpha: int = 0.05,
                         mode: str = 'central',
                         n_samples: int = 10000):
    assert len(targets) == len(predictions)
    assert mode in ['central', 'normal', 'percentiles']

    indexes = np.random.choice(len(targets), size=(n_samples, len(targets)), replace=True)
    bootstraped_targets, bootstraped_predictions = targets[indexes], predictions[indexes]
    if callable(metrics):
        metric = metrics
        estimate = metric(targets, predictions)
        bootstraped_metric = get_bootstraped_metric(bootstraped_targets, bootstraped_predictions,
                                                    metric=metric)
        (left, right) = get_confidence_intervals(bootstraped_metric, estimate,
                                                 alpha=alpha, mode=mode)
        return estimate, (left, right)
    if isinstance(metrics, list):
        result = []
        for metric in metrics:
            estimate = metric(targets, predictions)
            bootstraped_metric = get_bootstraped_metric(bootstraped_targets,
                                                        bootstraped_predictions,
                                                        metric=metric)
            (left, right) = get_confidence_intervals(bootstraped_metric, estimate,
                                                     alpha=alpha, mode=mode)
            result.append((estimate, (left, right)))
        return result
    if isinstance(metrics, dict):
        result = {}
        for label, metric in metrics.items():
            estimate = metric(targets, predictions)
            bootstraped_metric = get_bootstraped_metric(bootstraped_targets,
                                                        bootstraped_predictions,
                                                        metric)
            (left, right) = get_confidence_intervals(bootstraped_metric, estimate,
                                                     alpha=alpha, mode=mode)
            result[label] = (estimate, (left, right))
    return result
