import argparse
import traceback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import utils.callbacks
import utils.data
import utils.email
import utils.logging

DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
    "argentina": {"feat": "data/2013_2014_argentina_y.csv", "adj": "data/2013_2014_argentina_adj.csv"},
    "brazil": {"feat": "data/2013_2014_brazil_y.csv", "adj": "data/2013_2014_brazil_adj.csv"},
    "chile": {"feat": "data/2013_2014_chile_y.csv", "adj": "data/2013_2014_chile_adj.csv"},
    "colombia": {"feat": "data/2013_2014_colombia_y.csv", "adj": "data/2013_2014_colombia_adj.csv"},
    "mexico": {"feat": "data/2013_2014_mexico_y.csv", "adj": "data/2013_2014_mexico_adj.csv"},
    "paraguay": {"feat": "data/2013_2014_paraguay_y.csv", "adj": "data/2013_2014_paraguay_adj.csv"},
    "uruguay": {"feat": "data/2013_2014_uruguay_y.csv", "adj": "data/2013_2014_uruguay_adj.csv"},
    "venezuela": {"feat": "data/2013_2014_venezuela_y.csv", "adj": "data/2013_2014_venezuela_adj.csv"},
    "flu_11_12": {"feat": "data/2011-2012_flu_normalized_y.csv", "adj": "data/2011-2012_flu_normalized_adj.csv"},
    "flu_13_14": {"feat": "data/2013-2014_flu_normalized_y.csv", "adj": "data/2013-2014_flu_normalized_adj.csv"},
    "china_air": {"feat": "data/china_air_y.csv", "adj": "data/china_air_adj.csv"}
}




def get_model(args, dm):
    model = None
    if args.model_name == "GCN":
        model = models.GCN(adj=dm.adj, input_dim=args.seq_len, output_dim=args.hidden_dim)
    if args.model_name == "GRU":
        model = models.GRU(input_dim=dm.adj.shape[0], hidden_dim=args.hidden_dim)
    if args.model_name == "TGCN":
        model = models.TGCN(adj=dm.adj, hidden_dim=args.hidden_dim)
    return model


def get_task(args, model, dm):
    task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
        model=model, feat_max_val=dm.feat_max_val, **vars(args)
    )
    return task


def get_callbacks(args):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="train_loss")
    plot_validation_predictions_callback = utils.callbacks.PlotValidationPredictionsCallback(monitor="train_loss")
    callbacks = [
        checkpoint_callback,
        plot_validation_predictions_callback,
    ]
    return callbacks


def main_supervised(args):
    dm = utils.data.SpatioTemporalCSVDataModule(
        feat_path=DATA_PATHS[args.data]["feat"], adj_path=DATA_PATHS[args.data]["adj"], **vars(args)
    )
    model = get_model(args, dm)
    task = get_task(args, model, dm)
    callbacks = get_callbacks(args)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(task, dm)
    results = trainer.validate(datamodule=dm)

    return results


def main(args):
    rank_zero_info(vars(args))
    results = globals()["main_" + args.settings](args)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--data", type=str, help="The name of the dataset", choices=("shenzhen", "losloop",
                                                                     "argentina", "brazil", "chile", "colombia",
                                                                     "mexico", "paraguay", "uruguay", "venezuela",
                                                                     "flu_11_12", "flu_13_14", "china_air" ), default="losloop"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("GCN", "GRU", "TGCN"),
        default="TGCN",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised",),
        default="supervised",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")
    # parser.add_argument("--send_email", "--email", action="store_true", help="Send email when finished")


    parser.add_argument("--num_class", type=int, default=2, help="Number of classes") # added by TC # civil 3, flu 5, air 5

    temp_args, _ = parser.parse_known_args()

    parser = getattr(utils.data, temp_args.settings.capitalize() + "DataModule").add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.capitalize() + "ForecastTask").add_task_specific_arguments(parser)

    args = parser.parse_args()
    utils.logging.format_logger(pl._logger)
    if args.log_path is not None:
        utils.logging.output_logger_to_file(pl._logger, args.log_path)

    results = main(args)
    # try:
    #     results = main(args)
    # except:  # noqa: E722
    #     traceback.print_exc()
    #     if args.send_email:
    #         tb = traceback.format_exc()
    #         subject = "[Email Bot][❌] " + "-".join([args.settings, args.model_name, args.data])
    #         utils.email.send_email(tb, subject)
    #     exit(-1)

    # if args.send_email:
    #     subject = "[Email Bot][✅] " + "-".join([args.settings, args.model_name, args.data])
    #     utils.email.send_experiment_results_email(args, results, subject=subject)
