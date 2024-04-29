import torch
import torch.nn as nn

from devtorch import train, DevModel

# TODO: This could be tested in a lot more detail


class MockModel(DevModel):
    def __init__(self, n_in=10, n_out=10):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(n_in, n_out))


class MockDataset:
    pass


def test_trainer():
    def loss(output, target):
        return F.cross_entropy(output, target.long())

    trainer = train.get_trainer(
        loss,
        model=MockModel(),
        train_dataset=MockDataset(),
        n_epochs=10,
        batch_size=128,
        lr=0.001,
        device="cpu",
        id="test",
    )
    assert trainer.n_epochs == 10
    assert trainer.batch_size == 128
    assert trainer.root == ""
    assert trainer.lr == 0.001
    assert trainer.optimizer_func == torch.optim.Adam
    assert trainer.scheduler_func == None
    assert trainer.device == "cpu"
    assert trainer.dtype == torch.float
    assert trainer.grad_clip_type == None
    assert trainer.grad_clip_value == None
    assert trainer.save_type == "SAVE_DICT"
    assert trainer.id == "test"
    assert trainer.optimizer_kwargs == {}
    assert trainer.scheduler_kwargs == {}
