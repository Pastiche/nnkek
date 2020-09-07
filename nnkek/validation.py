import torch


class TopK:
    def __init__(self, vectors: torch.Tensor, k=5):
        distance_mtrx = torch.cdist(vectors, vectors)
        self.distances, self.indices = torch.topk(distance_mtrx, k)

    def __getitem__(self, i: int):
        return self.indices[i]

    @property
    def shape(self):
        return self.indices.shape

    def __len__(self):
        return self.shape[0]


class EncoderValidator:
    def __init__(self, raw: torch.Tensor, encoded: torch.Tensor, k=5):
        self.raw = TopK(raw, k)
        self.encoded = TopK(encoded, k)

    def __getitem__(self, i: int):
        return self.raw[i], self.encoded[i]

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    @property
    def shape(self):
        return self.raw.shape

    def __len__(self):
        return self.shape[0]
