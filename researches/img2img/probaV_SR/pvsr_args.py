import argparse


def parse_arguments():
  parser = argparse.ArgumentParser(description='Textbox Detector Settings')
  ##############
  #        TRAINING        #
  ##############
  parser.add_argument(
    "--train",
    action="store_true",
    help="do train mode",
  )
  parser.add_argument(
    "--test",
    action="store_true",
    help="do test mode",
  )
  parser.add_argument(
    "-tr",
    "--trellis",
    action="store_true",
    help="use trellis, only when -wm is rdn",
  )
  parser.add_argument(
    "-hp",
    "--half_precision",
    action="store_true",
    help="use half precision mode to accelerate and lower memory consumption",
  )
  parser.add_argument(
    "-wm",
    "--which_model",
    type=str,
    help="choose between rdn, carn, basic",
    default="basic"
  )
  parser.add_argument(
    "-dt",
    "--deterministic_train",
    action="store_true",
    help="if this is turned on, then everything will be deterministic"
         "and the training process will be reproducible.",
  )
  parser.add_argument(
    "-en",
    "--epoch_num",
    type=int,
    help="Epoch number of the training",
    default=200
  )
  parser.add_argument(
    "--filters",
    type=int,
    help="Epoch number of the training",
    default=64
  )
  parser.add_argument(
    "-cv",
    "--cross_val",
    type=int,
    help="number of cross validation",
    default=1
  )
  parser.add_argument(
    "-bpg",
    "--batch_size_per_gpu",
    type=int,
    help="batch size inside each GPU during training",
    default=3
  )
  parser.add_argument(
    "-lt",
    "--loading_threads",
    type=int,
    help="loading_threads correspond to each GPU during both training and validation, "
         "e.g. You have 4 GPU and set -lt 2, so 8 threads will be used to load data",
    default=2
  )
  parser.add_argument(
      "-mp",
      "--model_prefix",
      type=str,
      help="prefix of model",
  )
  args = parser.parse_args()
  return args
