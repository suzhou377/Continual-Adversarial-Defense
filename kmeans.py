from functools import partial
import numpy as np
import torch
def initialize(X, num_clusters, seed):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param seed: (int) seed for kmeans
    :return: (np.array) initial state
    """
    num_samples = len(X)
    if seed == None:
        indices = np.random.choice(num_samples, num_clusters, replace=True)
    else:
        np.random.seed(seed)
        indices = np.random.choice(num_samples, num_clusters, replace=True)
    initial_state = X[indices]
    return initial_state
'''计算原型'''
def calculate_prototype(logit, cluster, unique_label, y):
    prototypes = torch.zeros(len(unique_label), logit.size(1)).to(logit.device)
    for index in unique_label:
        # 获取当前伪标签对应的样本索引
        selected = torch.where(cluster == index)[0]
        selected_logits = torch.index_select(logit, 0, selected)
        # 如果没有样本对应当前标签，则根据真实标签随机选择一个样本
        if selected_logits.shape[0] == 0:
            selected_indices = torch.where(y == index)[0]
            # 如果连真实标签都没有对应的样本，则随机选择一个样本
            if selected_indices.numel() == 0:
                selected_logits = logit[torch.randint(len(logit), (1,))]
            else:
                random_index = torch.randint(selected_indices.numel(), (1,)).item()
                selected_logits = logit[selected_indices[random_index].unsqueeze(0)]
        # 计算当前标签的原型向量
        prototypes[index] = selected_logits.mean(dim=0)
    return prototypes

def kmeans(
        X,
        num_clusters,
        unique_label,
        y,
        distance='euclidean',
        cluster_centers=[],
        tol=1e-4,
        iter_lmit=200,
        device=torch.device('cuda'),
        seed=None,
):
    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, device=device)
    elif distance == 'cosine':
        pairwise_distance_function = partial(pairwise_cosine, device=device)
    else:
        raise NotImplementedError
    X = X.float()
    X = X.to(device)
    if type(cluster_centers) == list:  # ToDo: make this less annoyingly weird
        initial_state = initialize(X, num_clusters, seed=seed)
    else:
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)

    iteration = 0

    while iteration <= iter_lmit:

        dis = pairwise_distance_function(X, initial_state)
        choice_cluster = torch.argmin(dis, dim=1)
        initial_state_pre = initial_state.clone()
        initial_state = calculate_prototype(X, choice_cluster, unique_label, y)
        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        iteration = iteration + 1

        if center_shift ** 2 < tol:
            break
        if iteration >= iter_lmit:
            break

    return choice_cluster, initial_state

def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cuda'),
        gamma_for_soft_dtw=0.001):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor) cluster ids
    """


    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, device=device)
    elif distance == 'cosine':
        pairwise_distance_function = partial(pairwise_cosine, device=device)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster


def pairwise_distance(data1, data2, device=torch.device('cuda')):

    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1).clone()

    # 1*N*M
    B = data2.unsqueeze(dim=0).clone()

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


