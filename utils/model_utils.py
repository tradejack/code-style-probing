import torch


def label_tensor_to_one_hot(label_tensor, style_dim):
    one_hot_tensor_list = []
    for idx in range(label_tensor.shape[0]):
        cluster_idx = label_tensor[idx].item()
        one_hot_tensor = cluster_to_one_hot_tensor(cluster_idx, style_dim)
        one_hot_tensor_list.append(one_hot_tensor.unsqueeze(0))

    return torch.cat(one_hot_tensor_list, dim=0)


def cluster_to_one_hot_tensor(cluster_idx, style_dim):
    style_tensor = torch.zeros(style_dim)
    if cluster_idx < 0:
        return style_tensor
    style_tensor[cluster_idx] = 1
    return style_tensor
