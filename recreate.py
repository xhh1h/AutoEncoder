import torch
from models import Encoder, Decoder
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder().to(device)
decoder = Decoder().to(device)
encoder.load_state_dict(torch.load("checkpoints/encoder_epoch3.pth", map_location=device))
decoder.load_state_dict(torch.load("checkpoints/decoder_epoch3.pth", map_location=device))
encoder.eval()
decoder.eval()

transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

img = Image.open("outputs/anonymized.png").convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    z = encoder(img_tensor)
    recon = decoder(z)

save_image(recon, "output_restored.png")
