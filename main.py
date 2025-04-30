import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from models import Encoder, Decoder
from dp_utils import add_dp_noise
from facenet_pytorch import MTCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)
decoder = Decoder().to(device)
encoder.load_state_dict(torch.load("checkpoints/encoder_epoch50.pth", map_location=device))
decoder.load_state_dict(torch.load("checkpoints/decoder_epoch50.pth", map_location=device))
encoder.eval()
decoder.eval()
def detect_faces(image):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(keep_all=True, device=device)
    boxes, _ = mtcnn.detect(image)
    return boxes
image_path = "test.jpg"
original_img = Image.open(image_path).convert("RGB")
image_np = np.array(original_img)
boxes = detect_faces(original_img)

if boxes is None:
    print("未检测到人脸")
    exit()

for box in boxes:
    x1, y1, x2, y2 = [int(b) for b in box]
    face_crop = original_img.crop((x1, y1, x2, y2))
    face_tensor = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])(face_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        z = encoder(face_tensor)
        z_noisy = add_dp_noise(z, epsilon=1.0, delta=1e-5)
        recon_face = decoder(z_noisy).squeeze(0).cpu()

    recon_img = transforms.ToPILImage()(recon_face).resize((x2 - x1, y2 - y1))
    image_np[y1:y2, x1:x2, :] = np.array(recon_img)

Image.fromarray(image_np).save("outputs/anonymized.png")