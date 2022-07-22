import random

import pytorch_lightning as pl
import torch
from torch.autograd import Variable


from utils.model_utils import label_tensor_to_one_hot
from data import cluster_labels_no_outliers, tokenizer


class PLGANModel(pl.LightningModule):
    def __init__(self, generator, discriminator, style_dim, device):
        super().__init__()
        # put the GAN model here
        self.generator = generator
        self.discriminator = discriminator
        self.style_dim = style_dim
        self.criterion = torch.nn.BCELoss()
        self.gpu_device = device

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        # update D network
        # All-real training
        # Format real batch
        real_data = batch["input_ids"]
        seq_len = real_data.shape[1]
        real_style = label_tensor_to_one_hot(
            batch["labels"], self.style_dim
        ).to(self.gpu_device)

        # Forward pass real batch through D
        output = self.discriminator(real_data)

        # Calculate loss on all-real batch
        discriminator_real_loss = self.criterion(output, real_style)

        # All-fake training

        # sampling the target styles for a whole patch
        sampled_style_indexes = random.sample(
            list(cluster_labels_no_outliers), real_data.shape[0]
        )
        style_encoding = label_tensor_to_one_hot(
            torch.Tensor(sampled_style_indexes).long(), self.style_dim
        ).to(self.gpu_device)

        # Forward pass to generate the styled output
        generator_output, modifier_output = self.generator(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            style_encoding=style_encoding,
        )
        generated_logits = generator_output.logits

        # use Gumbel Softmax to decode the output
        generated_tokens = torch.nn.functional.gumbel_softmax(
            generated_logits, hard=True, dim=-1
        )

        # produce the fake data
        fake_data = generated_tokens.argmax(-1)
        # print(tokenizer.batch_decode(fake_data))

        # Classify all fake batch with D
        output = self.discriminator(fake_data)

        false_label = torch.zeros(style_encoding.shape).to(self.gpu_device)

        # Calculate D's loss on the all-fake batch
        discriminator_fake_loss = self.criterion(output, false_label)

        # Compute error of D as sum over the fake and the real batches
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss

        self.log("d_loss", discriminator_loss)
        # Yield instead of return: This makes the training_step a Python generator.
        # Once we call it again, it will continue the execution with the block below
        yield Variable(discriminator_loss, requires_grad=True)

        # update M network
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.discriminator(fake_data)

        # Calculate G's loss based on this output
        generator_class_loss = self.criterion(output, style_encoding)

        generated_text_batch = tokenizer.batch_decode(
            fake_data, skip_special_tokens=True
        )
        generated_input_batch = tokenizer(
            generated_text_batch,
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        generated_input_batch = {
            k: v.to(self.gpu_device) for k, v in generated_input_batch.items()
        }
        generated_encoding = self.generator.get_encoding(
            **generated_input_batch
        )[0]
        modifier_loss = (
            1
            - torch.nn.CosineSimilarity(dim=-1, eps=1e-08)(
                generated_encoding, modifier_output
            ).mean()
        )

        generator_loss = generator_class_loss + modifier_loss

        self.log("g_loss", generator_loss)
        yield Variable(generator_loss, requires_grad=True)

    def validation_step(self, batch, batch_idx):
        # update D network
        # All-real training
        # Format real batch
        real_data = batch["input_ids"]
        seq_len = real_data.shape[1]
        real_style = label_tensor_to_one_hot(
            batch["labels"], self.style_dim
        ).to(self.gpu_device)

        # Forward pass real batch through D
        output = self.discriminator(real_data)

        # Calculate loss on all-real batch
        discriminator_real_loss = self.criterion(output, real_style)

        # All-fake training
        # sampling the target styles for a whole patch
        sampled_style_indexes = random.sample(
            list(cluster_labels_no_outliers), real_data.shape[0]
        )
        style_encoding = label_tensor_to_one_hot(
            torch.Tensor(sampled_style_indexes).long(), self.style_dim
        ).to(self.gpu_device)

        # Forward pass to generate the styled output
        generator_output, modifier_output = self.generator(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            style_encoding=style_encoding,
        )
        generated_logits = generator_output.logits

        # use Gumbel Softmax to decode the output
        generated_tokens = torch.nn.functional.gumbel_softmax(
            generated_logits, hard=True, dim=-1
        )

        # produce the fake data
        fake_data = generated_tokens.argmax(-1)
        # print(tokenizer.batch_decode(fake_data))

        # Classify all fake batch with D
        output = self.discriminator(fake_data)

        false_label = torch.zeros(style_encoding.shape).to(self.gpu_device)

        # Calculate D's loss on the all-fake batch
        discriminator_fake_loss = self.criterion(output, false_label)

        # Compute error of D as sum over the fake and the real batches
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss

        self.log("val_d_loss", discriminator_loss)
        # Yield instead of return: This makes the training_step a Python generator.
        # Once we call it again, it will continue the execution with the block below

        # update M network
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.discriminator(fake_data)

        # Calculate G's loss based on this output
        generator_class_loss = self.criterion(output, style_encoding)

        generated_text_batch = tokenizer.batch_decode(
            fake_data, skip_special_tokens=True
        )
        generated_input_batch = tokenizer(
            generated_text_batch,
            max_length=seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        generated_input_batch = {
            k: v.to(self.gpu_device) for k, v in generated_input_batch.items()
        }
        generated_encoding = self.generator.get_encoding(
            **generated_input_batch
        )[0]
        modifier_loss = (
            1
            - torch.nn.CosineSimilarity(dim=-1, eps=1e-08)(
                generated_encoding, modifier_output
            ).mean()
        )
        generator_loss = generator_class_loss + modifier_loss

        self.log("val_g_loss", generator_loss)

    # def test_step(self, batch, batch_idx):

    def configure_optimizers(self):
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters()
        )
        generator_optimizer = torch.optim.Adam(self.generator.parameters())

        return [discriminator_optimizer, generator_optimizer], []
