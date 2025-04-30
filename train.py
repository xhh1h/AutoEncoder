import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from models import Encoder, Decoder
from PIL import Image

latent_dim = 128
epochs = 50
batch_size = 64
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_root = './data'
# 自定义 CelebA 数据集（本地版）
class CustomCelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'celeba', 'img_align_celeba')

        if not os.path.exists(self.image_dir):
            raise RuntimeError(f"Image directory not found at {self.image_dir}")

        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.image_files.sort()
        print(f"Found {len(self.image_files)} images in the dataset.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # dummy label


def main():
    os.makedirs(dataset_root, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_dataset = CustomCelebADataset(root_dir=dataset_root, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    encoder = Encoder(latent_dim).to(device)
    decoder = Decoder(latent_dim).to(device)

    criterion = nn.MSELoss()
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)

    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        epoch_loss = 0

        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad()
            z = encoder(images)
            recon = decoder(z)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"[Epoch {epoch}/{epochs}] Loss: {epoch_loss / len(train_loader):.4f}")
        torch.save(encoder.state_dict(), f"checkpoints/encoder_epoch{epoch}.pth")
        torch.save(decoder.state_dict(), f"checkpoints/decoder_epoch{epoch}.pth")


if __name__ == '__main__':
    main()

