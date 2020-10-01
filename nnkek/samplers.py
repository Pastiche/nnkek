from torch.utils.data import Sampler
import numpy as np


class StratifiedConsequentBatchSampler(Sampler):
    # todo: expand for not only one sample from each strat, but distribution/weights, batches of batches
    def __init__(self, strata_sampler, batch_size=64, drop_last=True):
        self.strata_sampler = strata_sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for strata_index in self.strata_sampler:
            strata = self.stratas[strata_index]
            sample_index = np.random.choice(strata)
            batch.append(sample_index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.strata_sampler) // self.batch_size
        else:
            return (len(self.strata_sampler) + self.batch_size - 1) // self.batch_size


class OneVsBatchOfOthersSampler(Sampler):
    def __init__(self, sampler, stratas, batch_size=64):
        self.batch_size = batch_size
        self.stratas = stratas
        self.sampler = sampler

    def __iter__(self):
        seen = set()

        for current_index in self.sampler:
            if current_index in seen:
                continue
            batch = [current_index]
            augmenting_indices = self._get_augmenting_indices(self.stratas[current_index])
            batch.extend(augmenting_indices)
            yield batch
            seen.update(batch)

    def __len__(self):
        return len(self.sampler)

    def _get_augmenting_indices(self, exclude_strata):
        augmenting_stratas = set()
        augmenting_batch = []
        while len(augmenting_batch) < self.batch_size - 1:
            index = np.random.randint(0, len(self.stratas))
            if self.stratas[index] == exclude_strata:
                continue
            if self.stratas[index] in augmenting_stratas:
                continue
            augmenting_batch.append(index)
            augmenting_stratas.add(self.stratas[index])

        return augmenting_batch


# Подход ниже - с взятием рандомных N категорий для батча (без возврата) и взятием по одному айтему из каждой - и так пока все категории не переберем,
# - не подходит, потому что у нас мало эпох (треним на ЦПУ) => за 100 эпох в некоторых категориях модель и половины данных ни разу не видела
# from torch.utils.data import Sampler
# class StratifiedBatchSampler(Sampler):
#     # todo: expand for not only one sample from each strat, but distribution/weights, batches of batches
#     def __init__(self, stratas, batch_size=64, drop_last=True):
#         self.stratas = stratas
#         self.batch_size = batch_size
#         self.drop_last = drop_last

#     def __iter__(self):
#         batch = []
#         for strata in np.random.choice(self.stratas, size = self.batch_size, replace = True):
#             sample_index = np.random.choice(strata)
#             batch.append(sample_index)
#             if len(batch) == self.batch_size:
#                 yield batch
#                 batch = []
#         if len(batch) > 0 and not self.drop_last:
#             yield batch

#     def __len__(self):
#         # TBD
#         l = 0
#         for strata in self.stratas:
#             l+= len(strata)

#         if self.drop_last:
#             return l // self.batch_size
#         else:
#             return (l + self.batch_size - 1) // self.batch_size


# from torch.utils.data import Sampler, RandomSampler
# class StratifiedBatchSampler(Sampler):
#     # todo: expand for not only one sample from each strat, but distribution/weights, batches of batches
#     def __init__(self, stratas, sampler, batch_size=64, drop_last=True):
#         self.stratas = stratas
#         self.sampler = sampler
#         self.batch_size = batch_size
#         self.drop_last = drop_last

#     def __iter__(self):
#         batch = []
#         for strata_index in self.sampler:
#             strata = self.stratas[strata_index]
#             sample_index = np.random.choice(strata)
#             batch.append(sample_index)
#             if len(batch) == self.batch_size:
#                 yield batch
#                 batch = []
#         if len(batch) > 0 and not self.drop_last:
#             yield batch

#     def __len__(self):
#         if self.drop_last:
#             return len(self.sampler) // self.batch_size
#         else:
#             return (len(self.sampler) + self.batch_size - 1) // self.batch_size

# from torch.utils.data import Sampler, RandomSampler
# class UniqueCateSampler(Sampler):
#     def __init__(self, sampler, batch_size=64, drop_last=True):
#         self.sampler = sampler
#         self.batch_size = batch_size
#         self.drop_last = drop_last
