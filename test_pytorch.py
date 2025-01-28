import torch


def test_torch():
    x = torch.rand(5, 3)
    assert x.shape == (5, 3)
    print("todo ok")
    if torch.cuda.is_available():
        print("cuda is available")


if __name__ == "__main__":
    print(torch.__version__)
    print(torch.version.cuda)
    test_torch()
