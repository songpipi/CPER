import torch
from torch import nn
import pickle
import ipdb

class AlignmentLevelBucket(nn.Module):
    def __init__(self, bucket):
        super().__init__()
        self.bucket = bucket
        boundary = torch.tensor(self.bucket.bin_edges_[0][1:-1]).float()
        bin_center = torch.tensor(
            (self.bucket.bin_edges_[0][1:] + self.bucket.bin_edges_[0][:-1]) / 2
        ).float()

        self.register_buffer("boundary", boundary)
        self.register_buffer("bin_center", bin_center)

    def forward(self, x):
        return torch.bucketize(x, self.boundary, right=True)

    def inverse_transform(self, x):
        return self.bin_center[x]




class ControlEncoder(nn.Module):
    def __init__(self, bucket_path,control_signal_at_inference, dec_emb_size):
        super().__init__()
        with open(bucket_path, "rb") as f:
            bucket_data = pickle.load(f)
        self.bucket = AlignmentLevelBucket(bucket_data)
        self.num_bins = bucket_data.n_bins_[0]
        self.control_embedding = nn.Embedding(self.num_bins, dec_emb_size)
        self.register_buffer(
            "control_signal_at_inference",
            torch.tensor([control_signal_at_inference], dtype=torch.long),
        )


    def forward(self, bsz, clip_sim, stage):
        if stage == 'eval':
            # we use given control signal value
            control_signal = self.control_signal_at_inference.repeat(bsz)
        elif stage == 'test':
            control_signal = clip_sim.repeat(bsz)
        else:
            # at training time, we compute alignment level with bucketing on CLIP similarity
            control_signal = self.bucket(clip_sim)

        # ipdb.set_trace()
        control_emb = self.control_embedding(control_signal).squeeze(1)
        return control_emb



