import unittest

import numpy as np
from torch_geometric.data import Data, InMemoryDataset, DataLoader

from clique_finding_models.output_transforms import *

GRAPHS = [
    # A house graph.
    Data(
        x=torch.ones([5, 1]),
        edge_index=torch.tensor([
            [0, 0, 1, 1, 1, 2, 2, 3, 1, 2, 2, 3, 4, 3, 4, 4],
            [1, 2, 2, 3, 4, 3, 4, 4, 0, 0, 1, 1, 1, 2, 2, 3]], dtype=torch.long), 
        y=torch.tensor([3, 4, 4, 4, 4], dtype=torch.float),
    ),
    # A line graph and a node
    Data(
        x=torch.ones([7, 1]),
        edge_index=torch.tensor([
            [0, 1, 2, 3, 4, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 0, 1, 2, 3, 4]], dtype=torch.long),
        y=torch.tensor([2, 2, 2, 2, 2, 2, 1], dtype=torch.float),
    ),
]


def clone(data):
    return Data.from_dict({k: v.clone() for k, v in data})


class TestDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TestDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = self.collate(GRAPHS)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass


class TestTransformY(object):
    def setUp(self):
        self.graphs = [clone(g) for g in GRAPHS]
        self.transform_y = None

    def test_reverse(self):
        for i in range(len(GRAPHS)):
            g = self.transform_y(self.graphs[i])
            y = self.transform_y.reverse(g, g.y)
            np.testing.assert_allclose(y.numpy(), GRAPHS[i].y.numpy())

    def test_reverse_batch_1(self):
        data = TestDataset(root="", transform=self.transform_y)
        loader = DataLoader(data, batch_size=1, shuffle=False)
        for i, batch in enumerate(loader):
            y = self.transform_y.reverse(batch, batch.y)
            np.testing.assert_allclose(y.numpy(), GRAPHS[i].y.numpy())

    def test_reverse_batch_2(self):
        data = TestDataset(root="", transform=self.transform_y)
        loader = DataLoader(data, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        y = self.transform_y.reverse(batch, batch.y)
        np.testing.assert_allclose(y.numpy(), torch.cat([GRAPHS[0].y, GRAPHS[1].y]) .numpy())


class TestTransformYNone(TestTransformY, unittest.TestCase):
    def setUp(self):
        super(TestTransformYNone, self).setUp()
        self.transform_y = transform_y_none

    def test_transform(self):
        g0 = transform_y_none(self.graphs[0])
        np.testing.assert_allclose(g0.y.numpy(), GRAPHS[0].y.numpy())

        g1 = transform_y_none(self.graphs[1])
        np.testing.assert_allclose(g1.y.numpy(), GRAPHS[1].y.numpy())


class TestTransformYRelativeToMaxClique(TestTransformY, unittest.TestCase):
    def setUp(self):
        super(TestTransformYRelativeToMaxClique, self).setUp()
        self.transform_y = transform_y_relative_to_max_clique

    def test_transform(self):
        g0 = transform_y_relative_to_max_clique(self.graphs[0])
        np.testing.assert_allclose(g0.y.numpy(), [0.75, 1, 1, 1, 1])

        g1 = transform_y_relative_to_max_clique(self.graphs[1])
        np.testing.assert_allclose(g1.y.numpy(), [1, 1, 1, 1, 1, 1, 0.5])


class TestTransformYRelativeToDegree(TestTransformY, unittest.TestCase):
    def setUp(self):
        super(TestTransformYRelativeToDegree, self).setUp()
        self.transform_y = transform_y_relative_to_degree

    def test_transform(self):
        g0 = transform_y_relative_to_degree(self.graphs[0])
        np.testing.assert_allclose(g0.y.numpy(), [1, 4. / 5, 4. / 5, 1, 1])

        g1 = transform_y_relative_to_degree(self.graphs[1])
        np.testing.assert_allclose(g1.y.numpy(), [1, 2. / 3, 2. / 3, 2. / 3, 2. / 3, 1, 1])


if __name__ == '__main__':
    unittest.main()
