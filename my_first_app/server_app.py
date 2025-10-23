"""my-first-app: A Flower / PyTorch app."""

import torch
from .custom_strategy import CustomFedAdagrad
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.common import Metrics
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from my_first_app.task import Net


def custom_metrics_aggregation_fn(
    records: list[RecordDict], weighting_metric_name: str
) -> MetricRecord:
    """Extract the minimum value for each metric key."""
    aggregated_metrics = MetricRecord()

    # Track current minimum per key in a plain dict,
    # then copy into MetricRecord at the end
    mins = {}

    for record in records:
        for record_item in record.metric_records.values():
            for key, value in record_item.items():
                if key == weighting_metric_name:
                    # We exclude the weighting key from the aggregated MetricRecord
                    continue

                if key in mins:
                    if value < mins[key]:
                        mins[key] = value
                else:
                    mins[key] = value

    for key, value in mins.items():
        aggregated_metrics[key] = value

    return aggregated_metrics


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = CustomFedAdagrad(fraction_train=fraction_train,
                                evaluate_metrics_aggr_fn=custom_metrics_aggregation_fn,)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,


    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
