import os
import pickle
from os import path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def idx_to_episode_index(idx, index_data=None):
    """
    Converts an index in [0...len(dataset)] to an episode and step index (where step index
    is the index of the step in the specific episode)
    :param idx:
    :return: episode_idx, step_idx

    >>> index = {0: {"data": [1,2,3,4], "data_length": 4}, 1: {"data": [5,6,7,8], "data_length": 4}}
    >>> idx_to_episode_index(0, index)
    (0, 0)
    >>> idx_to_episode_index(2, index)
    (0, 2)
    >>> idx_to_episode_index(7, index)
    (1, 3)
    >>> idx_to_episode_index(8, index)
    Traceback (most recent call last):
    ...
    ValueError: Invalid index 8
    """
    assert index_data is not None

    total_steps = 0
    for episode, data in index_data.items():
        episode_length = data["data_length"]
        if total_steps + episode_length > idx:
            result_episode = episode
            result_step = idx - total_steps
            return result_episode, result_step
        total_steps += episode_length
    raise ValueError("Invalid index {}".format(idx))


"""
Dataset wrapping observations from previously recorded episodes.

Expects the data to be stored in .npz files in folder ./data
The ./data folder contains one folder per episode, which contains a episode_{episode_num}.npz file
"""


class ObsDataset(Dataset):

    def __init__(self, root_folder):
        """

        :param device:
        :param root_folder:

        >>> obs_dataset = ObsDataset("/home/dobrusii/Thesis/code/GymGrasping/experiments/002_record_dataset_from_policy/dataset_sim_default-domain-randomization/2020-02-25-17-03_Mini")
        >>> len(obs_dataset)
        46766
        >>> obs_dataset.__getitem__(0)
        []
        """
        self.root_folder = root_folder
        self.data_dir = os.listdir(os.path.join(self.root_folder, "data"))

        # Read folder containing data
        dirs = sorted(self.data_dir)

        self.index = {}

        # Iterate over all episodes and append to data
        self.total_length = 0
        for i, dir in enumerate(tqdm(dirs, desc="Creating episode index", unit='episodes')):
            episode_data_path = os.path.join(self.root_folder, "data", dir, dir + ".pickle")

            if not path.exists(episode_data_path):
                # Make sure pickle file already exists - when using dataset during data generation
                # it is possible that the episode folder is already created but no pickle file present.
                continue

            with open(episode_data_path, 'rb') as pickle_file:
                episode_data = pickle.load(pickle_file)
                episode_length = episode_data["steps"]

                self.index[i] = {"data_path": episode_data_path,
                                 "data_length": episode_length}
                self.total_length += episode_length

    def __len__(self):
        """
        Returns the total number of images in the dataset
        :return:
        """
        return self.total_length

    def __getitem__(self, idx):
        """
        Returns the item at index idx. Indices are calculated by counting all steps in the episodes
        over all episodes.
        :param idx:
        :return:
        """
        episode_idx, step_idx = idx_to_episode_index(idx, self.index)
        episode = self._load_episode_with_index(episode_idx)
        step_obs = episode["obs"][step_idx]

        for key, array in step_obs.items():
            step_obs[key] = torch.from_numpy(
                step_obs[key].squeeze())  # Squeeze to get rid of batch dim
        return step_obs

    def _load_episode_with_index(self, episode_index):
        """
        Loads a specific episode from disc for further processing
        :return:
        """
        episode_data_path = self.index[episode_index]["data_path"]
        with open(episode_data_path, 'rb') as pickle_file:
            episode_data = pickle.load(pickle_file)
            episode_data["data_path"] = episode_data_path
        return episode_data
