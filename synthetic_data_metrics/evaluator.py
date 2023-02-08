"""The core class for evaluating datasets."""

from copy import deepcopy
from . import metrics


class Inception_Evaluator:
    """The central class in `synthgauge`, used to hold and evaluate data
    via metrics and visualisation.
    Parameters
    ----------
    real : pandas.DataFrame
        Dataframe containing the real data.
    synth : pandas.DataFrame
        Dataframe containing the synthetic data.
    handle_nans : str, default "drop"
        Whether to drop missing values. If yes, use "drop" (default).
    Returns
    -------
    synthgauge.Evaluator
        An `Evaluator` object ready for metric and visual evaluation.
    """

    def __init__(self, synth):

        # Metrics is private to apply some validation
        self.__metrics = dict()  # assign metrics and kwargs
        self.metric_results = dict()  # store results

        self.synth_data = synth

    def metrics(self):
        """Return __metrics."""

        return self.__metrics
    
    def add_metric(self, name, **kwargs):
        """Add a metric to the evaluator.
        Metrics and their arguments are recorded to be run at a later
        time. This allows metric customisation but ensures that the same
        metric configuration is applied consistently, i.e. once added,
        the parameters do not require resupplying for each execution of
        the metric. Supplying a metric alias allows the same metric to
        be used multiple times with different parameters.
        Note that `self.real_data` and `self.synth_data` will be passed
        automatically to metrics that expect these arguments. They
        should not be declared in `metric_kwargs`.
        Parameters
        ----------
        name : str
            Name of the metric. Must match one of the functions in
            `synthgauge.metrics`.

        **kwargs : dict, optional
            Keyword arguments for the metric. Refer to the associated
            metric documentation for details.
        """

        try:
            getattr(metrics, name)
            kwargs["name"] = name
            self.__metrics.update({name: kwargs})

        except AttributeError:
            raise NotImplementedError(
                f"Metric '{name}' is not implemented")

        

    def evaluate(self, as_df=False):
        """Compute metrics for real and synth data.
        Run through the metrics dictionary and execute each with its
        corresponding arguments. The results are returned as either a
        dictionary or dataframe.
        Results are also silently stored as a dictionary in
        `self.metric_results`.
        Parameters
        ----------
        as_df : bool, default False
            If `True`, the results will be returned as a
            `pandas.DataFrame`, otherwise a dictionary is returned.
            Default is `False`.
        Returns
        -------
        pandas.DataFrame
            If `as_df` is `True`. Each row corresponds to a metric-
            value pair. Metrics with multiple values have multiple
            rows.
        dict
            If `as_df` is `False`. The keys are the metric names and
            the values are the metric values (grouped). Metrics with
            multiple values are assigned to a single key.
        """

        results = dict.fromkeys(self.__metrics.keys())
        print(results)
        
        metrics_copy = deepcopy(self.__metrics)
        for metric, kwargs in metrics_copy.items():
            metric_name = kwargs.pop("name")
            if metric_name in metrics.__dict__.keys():
                metric_func = getattr(metrics, metric_name)
            else:
                metric_func = kwargs.pop("func")
            results[metric] = metric_func(self.synth_data, **kwargs
            )

        self.metric_results = dict(results)

        if as_df:
            tidy_results = {}
            for k, v in self.metric_results.items():
                try:
                    for vk, vv in v._asdict().items():
                        tidy_results[k + "-" + vk] = vv
                except AttributeError:
                    tidy_results[k] = v
