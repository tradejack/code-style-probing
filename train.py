import random
import time

import torch
from tqdm.auto import tqdm

from model import InRepPlusGAN, Discriminator
from data import (
    STYLE_DIM,
    get_data_loader,
    cluster_labels_no_outliers,
    train_dataset,
    test_dataset,
)
from config import NUM_EPOCHS, BATCH_SIZE, NAME
from utils.model_utils import label_tensor_to_one_hot
from gpu import get_device

device, _ = get_device()

checkpoint_path = f"checkpoint/{NAME}_{round(time.time())}"


def training_step(
    input_batch,
    discriminator,
    generator,
    criterion,
    discriminator_optimizer,
    generator_optimizer,
):
    # update D network
    discriminator.zero_grad()

    # All-real training
    # Format real batch
    real_data = input_batch["input_ids"]
    real_style = label_tensor_to_one_hot(input_batch["labels"], STYLE_DIM).to(
        device
    )

    # Forward pass real batch through D
    output = discriminator(real_data)

    # Calculate loss on all-real batch
    discriminator_real_loss = criterion(output, real_style)

    # Calculate gradients for D in backward pass
    discriminator_real_loss.backward()

    # All-fake training

    # sampling the target styles for a whole patch
    sampled_style_indexes = random.sample(
        list(cluster_labels_no_outliers), BATCH_SIZE
    )
    style_encoding = label_tensor_to_one_hot(
        torch.Tensor(sampled_style_indexes).long(), STYLE_DIM
    ).to(device)

    # Forward pass to generate the styled output
    generator_output, modifier_output = generator(
        input_ids=input_batch["input_ids"],
        attention_mask=input_batch["attention_mask"],
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
    output = discriminator(fake_data)

    # Calculate D's loss on the all-fake batch
    discriminator_fake_loss = criterion(output, style_encoding)

    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    discriminator_fake_loss.backward()

    # Compute error of D as sum over the fake and the real batches
    discriminator_loss = discriminator_real_loss + discriminator_fake_loss

    # Update D
    discriminator_optimizer.step()

    # update M network
    generator.zero_grad()

    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = discriminator(fake_data)

    # Calculate G's loss based on this output
    generator_class_loss = criterion(output, style_encoding)

    # Calculate gradients for G
    generator_class_loss.backward()

    # TODO: add the modifier loss
    generator_loss = generator_class_loss

    # Update G
    generator_optimizer.step()

    return generator_loss, discriminator_loss


def train():
    train_loader = get_data_loader(train_dataset)
    test_loader = get_data_loader(test_dataset)
    generator = InRepPlusGAN(style_dim=STYLE_DIM).to(device)
    discriminator = Discriminator(
        vocab_size=generator.config.vocab_size,
        embedding_dim=512,
        output_size=128,
        style_dim=STYLE_DIM,
        device=device,
    ).to(device)
    # Initialize BCELoss function
    criterion = torch.nn.BCELoss()
    # Setup Adam optimizers for both G and D
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters())
    generator_optimizer = torch.optim.Adam(generator.parameters())
    for epoch in range(NUM_EPOCHS):
        epoch_loss_g = 0
        epoch_loss_d = 0
        for batch in tqdm(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss_g, loss_d = training_step(
                batch,
                discriminator,
                generator,
                criterion,
                discriminator_optimizer,
                generator_optimizer,
            )
            epoch_loss_g += loss_g
            epoch_loss_d += loss_d
        epoch_loss_g /= len(train_dataset)
        epoch_loss_d /= len(train_dataset)
        torch.save(
            {
                "epoch": epoch,
                "g_state_dict": generator.state_dict(),
                "d_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": generator_optimizer.state_dict(),
                "optimizer_d_state_dict": discriminator_optimizer.state_dict(),
                "g_loss": epoch_loss_g,
                "d_loss": epoch_loss_d,
            },
            f"{checkpoint_path}_epoch_{epoch}.pt",
        )


if __name__ == "__main__":
    train()
