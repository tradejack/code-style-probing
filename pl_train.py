import time
import inspect
from functools import partial

import pytorch_lightning as pl
from pytorch_lightning.loops import OptimizerLoop
from pytorch_lightning.loops.optimization.optimizer_loop import ClosureResult
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from model import InRepPlusGAN, Discriminator
from data import (
    STYLE_DIM,
    get_data_loader,
    train_dataset,
    test_dataset,
)
from config import NUM_EPOCHS, NAME
from gpu import get_device
from pl_model import PLGANModel

device, _ = get_device()

checkpoint_path = f"checkpoint/{NAME}_{round(time.time())}"


from pytorch_lightning.loops import OptimizerLoop


class YieldLoop(OptimizerLoop):
    def __init__(self):
        super().__init__()
        self._generator = None

    def connect(self, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not connect any child loops."
        )

    def on_run_start(self, split_batch, optimizers, kwargs):
        super().on_run_start(split_batch, optimizers, kwargs)
        if not inspect.isgeneratorfunction(
            self.trainer.lightning_module.training_step
        ):
            raise MisconfigurationException(
                "The `LightningModule` does not yield anything in the `training_step`."
            )
        assert self.trainer.lightning_module.automatic_optimization

        # We request the generator once and save it for later so we can call next() on it.
        self._generator = self._get_generator(split_batch, kwargs)

    def _make_step_fn(self, *_):
        return partial(self._training_step, self._generator)

    def _get_generator(self, split_batch, kwargs, opt_idx=0):
        # print(kwargs)
        # kwargs = self._build_kwargs(kwargs, opt_idx, hiddens=None)

        # Here we are basically calling `lightning_module.training_step()`
        # and this returns a generator! The `training_step` is handled by
        # the accelerator to enable distributed training.
        return self.trainer.strategy.training_step(split_batch, kwargs)

    def _training_step(self, generator):
        # required for logging
        self.trainer.lightning_module._current_fx_name = "training_step"

        # Here, instead of calling `lightning_module.training_step()`
        # we call next() on the generator!
        training_step_output = next(generator)
        self.trainer.strategy.post_training_step()

        model_output = self.trainer._call_lightning_module_hook(
            "training_step_end", training_step_output
        )
        strategy_output = self.trainer._call_strategy_hook(
            "training_step_end", training_step_output
        )
        training_step_output = (
            strategy_output if model_output is None else model_output
        )

        # The closure result takes care of properly detaching the loss for logging and peforms
        # some additional checks that the output format is correct.
        result = ClosureResult.from_training_step_output(
            training_step_output, self.trainer.accumulate_grad_batches
        )
        return result


def train():
    train_loader = get_data_loader(train_dataset)
    test_loader = get_data_loader(test_dataset)
    generator = InRepPlusGAN(style_dim=STYLE_DIM).to(device)
    discriminator = Discriminator(
        vocab_size=generator.config.vocab_size,
        embedding_layer=generator.encoder.embed_tokens,
        embedding_dim=generator.config.d_model,
        output_size=768,
        style_dim=STYLE_DIM,
        device=device,
    ).to(device)
    # init model
    gan_model = PLGANModel(
        generator=generator,
        discriminator=discriminator,
        style_dim=STYLE_DIM,
        device=device,
    )

    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=NUM_EPOCHS,
        progress_bar_refresh_rate=20,
        default_root_dir=d ,
    )

    trainer.fit_loop.epoch_loop.batch_loop.connect(optimizer_loop=YieldLoop())
    # Train the model ⚡
    trainer.fit(gan_model, train_loader, test_loader)


if __name__ == "__main__":
    train()
