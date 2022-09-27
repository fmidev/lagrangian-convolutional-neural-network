from abc import ABC, abstractmethod, abstractstaticmethod

class Metric(ABC):

    def __init__(self):
        """
        Should contain:
            initialization of contingency tables as self.tables attribute,
            other parameters,
            is_empty flag, 
            name_template
        """
        pass
    
    @abstractmethod
    def accumulate(self, x_pred, x_obs):
        """
        Accumulates self.tables using predictions x_pred and observations x_obs
        """
        pass

    @abstractmethod
    def compute(self):
        """
        Computes final metrics from self.tables and returns array of metric values and 
        list of metric names. 
        """
        pass

    @abstractmethod
    def merge(self, other):
        """
        Merge contingency table of another metric of the same type to this metric object.
        """
        pass

    @abstractstaticmethod
    def plot(metric_representation, fig, **kwargs):
        """
        Plot representation of the metric to 'fig', or create a new figure.
        A matplotlib.pyplot.Figure object is returned
        """
        return fig
