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
    tokenizer,
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
    seq_len = real_data.shape[1]
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
        list(cluster_labels_no_outliers), real_data.shape[0]
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

    false_label = torch.zeros(style_encoding.shape).to(device)

    # Calculate D's loss on the all-fake batch
    discriminator_fake_loss = criterion(output, false_label)

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
        k: v.to(device) for k, v in generated_input_batch.items()
    }
    generated_encoding = generator.get_encoding(**generated_input_batch)[0]
    modifier_loss = (
        1
        - torch.nn.CosineSimilarity(dim=-1, eps=1e-08)(
            generated_encoding, modifier_output
        ).mean()
    )

    modifier_loss.backward()

    generator_loss = generator_class_loss + modifier_loss

    # Update G
    generator_optimizer.step()

    return generator_loss, discriminator_loss


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
