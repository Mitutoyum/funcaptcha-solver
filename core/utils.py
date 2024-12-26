from torch.nn import functional as F


def calculate_distance(x1, x2):
    return F.pairwise_distance(x1, x2)
