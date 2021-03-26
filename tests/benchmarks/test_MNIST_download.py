import unittest

from torchvision import datasets


import tempfile

from dacbench.envs.sgd import set_header_for


class TestMNISTDownload(unittest.TestCase):
    """
    Just for separately testing if downloading of MNIST is working properly
    """

    def test_download_old(self):
        set_header_for(
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "train-images-idx3-ubyte.gz",
        )
        set_header_for(
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
        )
        set_header_for(
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
        )
        set_header_for(
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            train_dataset = datasets.MNIST(temp_dir, train=True, download=True)
            assert train_dataset is not None

    def test_download(self):
        import urllib

        opener = urllib.request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)

        with tempfile.TemporaryDirectory() as temp_dir:
            train_dataset = datasets.MNIST(temp_dir, train=True, download=True)
            assert train_dataset is not None
