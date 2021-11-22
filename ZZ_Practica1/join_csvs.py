from argparse import ArgumentParser

import pandas as pd

parser = ArgumentParser()
parser.add_argument("-o", "--output", dest="output_file",
                    help="write result CSV to FILE", metavar="FILE",
                    default="data/train.csv")
args = parser.parse_args()

color_data = pd.read_csv("data/train/color_train.csv").drop(columns="Class")
shape_data = pd.read_csv("data/train/shape_train.csv")
texture_data = pd.read_csv("data/train/texture_train.csv")

data = color_data.join(shape_data).join(texture_data)
data.to_csv(args.output_file, index=False)
