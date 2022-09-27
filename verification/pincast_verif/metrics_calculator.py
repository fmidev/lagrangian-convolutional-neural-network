import os
import shutil
import logging

from attrdict import AttrDict
import numpy as np
import pandas as pd
import dask

from  pincast_verif import metric_tools
from pincast_verif import io_tools
from pincast_verif.masking import mask


class MetricsCalculator(object):
    """
    Class providing an interface for metrics calculation for nowcasts

    Methods :
    # to call at the start of the script #
    - __init__()
        calls -> _init_contincency_tables()
        calls -> _init_done_df()
    - get_samples_left()

    # to call inside the main calculation loop #
    - accumulate()
    - save_done_df()
    - save_contingency_tables()

    # to call at the end of the script, once metrics are accumulated #
    - compute()
    - save_metrics()

    A usage example is given in scripts/calculate_metrics.py
    """

    def __init__(self, config: AttrDict, config_path: str = None) -> None:

        self.id = config.exp_id

        # init paths
        self.path = AttrDict()
        self.path.metrics = config.metrics_npy_path
        self.path.tables = config.tables_path
        self.path.timestamps = config.timestamps_path
        self.path.name = config.name_path
        self.path.done = config.done_csv_path

        # init config
        self.methods = config.methods
        self.measurements = config.measurements
        self.n_leadtimes = config.n_leadtimes
        self.leadtimes = range(1, self.n_leadtimes + 1)
        self.common_mask = config.common_mask
        self.verbose = config.verbose
        self.metrics = config.metrics
        self.metric_params = config.metric_params
        self.n_chunks = config.n_chunks
        self.n_workers = config.n_workers

        # init working lists and dicts
        self.outdict, self.has_loaded_table = self._init_contingency_tables(
            attempt_load=True
        )
        self.timestamps = io_tools.read_file(self.path.timestamps, start_idx=1)
        self.done_df = self._init_done_df()
        if config.debugging:
            import random

            random.seed(12345)
            self.samples_left = random.sample(
                population=self.timestamps, k=config.debugging
            )
        else:
            self.samples_left = self.get_samples_left()

        if self.n_chunks > 0:
            self.chunks = np.array_split(np.array(self.samples_left), self.n_chunks)

        # init directory structure
        for path in self.path:
            for method in self.methods:
                for metric in self.metrics:
                    path_formatted = self.path[path].format(
                        id=self.id, method=method, metric=metric
                    )
                    if self.verbose:
                        logging.info("Initializing path to:")
                        logging.info(path_formatted)
                    os.makedirs(os.path.dirname(path_formatted), exist_ok=True)

        # copy config to results folder if source path specified
        if config_path is not None:
            shutil.copyfile(
                src=config_path, dst=config.config_copy_path.format(id=self.id)
            )

    def _init_contingency_tables(self, attempt_load: bool) -> dict:
        outdict = {}
        has_loaded_table = False
        for method in self.methods:
            table_path = self.path.tables.format(method=method, id=self.id)
            if os.path.exists(table_path) and attempt_load:
                has_loaded_table = True
                logging.info(
                    "existing output file found, loading {}".format(table_path)
                )
                outdict[method] = np.load(table_path, allow_pickle=True).item()
                # postprocessing
                for metric in outdict[method]:
                    outdict[method][metric] = metric_tools.get_metric(
                        metric_name=metric,
                        leadtimes=self.leadtimes,
                        metric_params=self.metric_params,
                        tables=outdict[method][metric],
                    )
            else:
                outdict[method] = {}
                for metric in self.metrics:
                    outdict[method][metric] = metric_tools.get_metric(
                        metric_name=metric,
                        leadtimes=self.leadtimes,
                        metric_params=self.metric_params,
                        tables=None,
                    )
        return outdict, has_loaded_table

    def save_contingency_tables(self) -> None:
        for method in self.methods:
            out_path = self.path.tables.format(method=method, id=self.id)
            # convert tables of object to dict for saving
            dicts_to_save = {}
            for metric in self.outdict[method]:
                dicts_to_save.update({metric: self.outdict[method][metric].tables})
            np.save(out_path, arr=dicts_to_save)

    def _init_done_df(self) -> pd.DataFrame:
        if os.path.exists(self.path.done.format(id=self.id)) and self.has_loaded_table:
            done = pd.read_csv(
                self.path.done.format(id=self.id), sep=",", header=0, index_col=0
            )
            for method in self.methods:
                if method not in done:
                    done[method] = False
            if "bad" not in done:
                raise ValueError("bad column not found in existing CSV, aborting")
        else:
            done = pd.DataFrame(
                index=self.timestamps, columns=self.methods, dtype=bool, data=False
            )
            done["bad"] = False
        return done

    def save_done_df(self) -> None:
        self.done_df.to_csv(self.path.done.format(id=self.id))

    def get_samples_left(self) -> list:
        indices_set = set(self.timestamps)
        existing_set = set(
            [
                self.timestamps[i]
                for i in range(len(self.timestamps))
                if all(self.done_df.iloc[i])
            ]
        )
        samples_left = list(indices_set - existing_set)
        return samples_left

    def accumulate(self, sample, done_df, outdict):
        if self.verbose:
            logging.info("\n Sample {} ongoing \n Loading data".format(sample))

        true = io_tools.load_observations(
            db_path=self.measurements.path, sample=sample, leadtimes=self.leadtimes
        )
        if true is None:
            done_df.loc[sample]["bad"] = True
            logging.warn(
                f"Sample containing missing observation found, skipping sample {sample}"
            )
            return done_df, outdict
        true = io_tools.dBZ_to_rainrate(true)
        preds = io_tools.load_predictions(self.methods, sample, self.leadtimes)
        if isinstance(preds, str):
            done_df.loc[sample]["bad"] = True
            logging.warn(
                f"Sample containing missing prediction {preds} found, skipping sample {sample}"
            )
            return done_df, outdict

        if self.common_mask:
            preds = mask(predictions=preds, n_leadtimes=self.n_leadtimes)

        for method in self.methods:
            if self.verbose:
                logging.info(
                    "Accumulating metrics for {method} predictions".format(
                        method=method
                    )
                )
            pred = io_tools.dBZ_to_rainrate(preds[method])
            for metric in outdict[method]:
                if self.verbose:
                    logging.info(f"Metric {metric} ongoing...")
                outdict[method][metric].accumulate(x_pred=pred, x_obs=true)
            done_df.loc[sample][method] = True

        del true, preds

        return done_df, outdict

    def update_state(self, done_df, outdict):
        self.done_df = done_df
        self.outdict = outdict

    def accumulate_chunk(self, chunk_index):
        _done_df, _outdict = (
            self._init_done_df(),
            self._init_contingency_tables(attempt_load=False)[0],
        )
        for sample in self.chunks[chunk_index]:
            _done_df, _outdict = self.accumulate(
                sample=sample, done_df=_done_df, outdict=_outdict
            )
        logging.info(f"Done with chunk {chunk_index}")
        return _done_df, _outdict

    def merge_done_dfs(self):
        partial_done_dfs = [data[0] for data in self.chunked_data]
        if self.has_loaded_table:
            partial_done_dfs.insert(0, self.done_df)
        self.done_df = metric_tools.merge_boolean_df_list(partial_done_dfs)

    def merge_outdicts(self):
        partial_outdicts = [data[1] for data in self.chunked_data]
        if self.has_loaded_table:
            partial_outdicts.insert(0, self.outdict)
        self.outdict = metric_tools.merge_outdict_list(partial_outdicts)

    def compute(self) -> dict:
        metrics_data = {}
        for method in self.methods:
            metrics_data.update({method: {}})
            for metric in self.outdict[method]:
                if self.outdict[method][metric].is_empty:
                    raise ValueError(
                        f"Tried to compute metrics for \
                        an empty contingency table {self.outdict[method][metric]}."
                    )
                values, names = self.outdict[method][metric].compute()
                metrics_data[method].update({metric: (values, names)})

        return metrics_data

    def save_metrics(self, metrics_dict) -> None:
        for method, metric_i_dict in metrics_dict.items():
            for metric_i, (value, name) in metric_i_dict.items():
                npy_path = self.path.metrics.format(
                    id=self.id, method=method, metric=metric_i
                )
                name_path = self.path.name.format(
                    id=self.id, method=method, metric=metric_i
                )
                np.save(file=npy_path, arr=value)
                with open(name_path, "w") as name_file:
                    for n in name:
                        name_file.write(n + "\n")

    def parallel_accumulation(self):
        res = []
        for chunk_idx in range(self.n_chunks):
            # y = self.accumulate_chunk(chunk_index=chunk_idx)
            y = dask.delayed(self.accumulate_chunk)(chunk_index=chunk_idx)
            res.append(y)

        scheduler = "processes" if self.n_workers > 1 else "single-threaded"
        self.chunked_data = dask.compute(
            *res, num_workers=self.n_workers, scheduler=scheduler, traverse=False
        )
        self.merge_outdicts()
        self.merge_done_dfs()
