# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import argparse
import logging
import os
import pickle
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import json


def get_parser():
    parser = argparse.ArgumentParser(description="Arguments for getting the bucket file")

    # Arguments for constructing bucket file
    parser.add_argument("--num_bucket_bins", default=3, type=int)
    # parser.add_argument("--write_with_score_annotation", action="store_true")

    # Arguments for annotations
    parser.add_argument("--annotation_path", default='type2/cl_train.json', help="annotation file path")
    parser.add_argument("--output_dir", default="type2/", help="output annotation dir")

    # Arguments for loader
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--batch_size", default=256, type=int)

    return parser.parse_args()


def load_clip_model(model_file):
    clip_model = clip.load(model_file, device="cpu")[0]
    # freeze the parameters of CLIP
    for param in clip_model.parameters():
        param.requires_grad = False
    return clip_model


def main():
    # init cfg
    args = get_parser()

    gen_res = []

    with open(args.annotation_path, 'r') as infile:
        data = json.load(infile)

    for image_id, image_data in data.items():
        distribs = image_data.get('origin_emotion_distribution')
        distribs = [x for x in distribs if x != 0]
        gen_res.extend(distribs)
    res_cos_sim = np.array(gen_res)
 
    est = KBinsDiscretizer(n_bins=args.num_bucket_bins, encode="ordinal", strategy="kmeans")#uniform\'quantile' 或 'kmeans'
    est.fit(res_cos_sim.reshape(-1, 1))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    output_path = args.output_dir + f"bucket_{args.num_bucket_bins}bins_kmeans.pickle"
    logging.info(f"Writing results to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(est, f)

if __name__ == "__main__":
    main()