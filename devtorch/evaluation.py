import os
import sys
import logging

import torch
import pandas as pd

from devtorch import util, query

logger = logging.getLogger("validator")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def build_metric_df(
    root,
    model_loader,
    dataset,
    metric,
    model_ids=None,
    batch_size=128,
    device="cuda",
    dtype=torch.float,
    **kwargs,
):

    metric_list = []
    model_ids = os.listdir(root) if model_ids is None else model_ids

    for model_id in model_ids:
        logger.info(f"Computing metric for {model_id}...")
        try:
            model = query.load_model(root, model_id, model_loader, device, dtype)
            metric_scores = compute_metric(
                model, dataset, metric, batch_size, device, dtype, **kwargs
            )

            for batch_id, metric_score in enumerate(metric_scores):
                row = {
                    "model_id": model_id,
                    "batch_id": batch_id,
                    "metric_score": metric_score,
                }
                metric_list.append(row)

        except Exception as error:
            logger.error(f"Failed computing metric for {model_id}: {error}")

    return pd.DataFrame(metric_list).set_index("model_id")


def compute_metric(
    model, dataset, metric, batch_size=128, device="cuda", dtype=torch.float, **kwargs
):
    metric_list = []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size)

    model.to(device)

    with torch.no_grad():
        for data, target in data_loader:
            data = util.cast(data, device, dtype)
            target = util.cast(target, device, dtype)

            if len(kwargs) > 0:
                output = model(data, **kwargs)
            else:
                output = model(data)
            metric_value = metric(output, target)
            metric_list.append(metric_value)

    return metric_list
