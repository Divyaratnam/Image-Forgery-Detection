import torch
from torchvision.utils import save_image
import os

from gan import Generator, LATENT_DIM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained generator
generator = Generator().to(DEVICE)
generator.load_state_dict(
    torch.load("models/gan_generator.pth", map_location=DEVICE)
)
generator.eval()

# Output directory
output_dir = "gan/generated_images/fake"
os.makedirs(output_dir, exist_ok=True)

NUM_IMAGES = 50

with torch.no_grad():
    for i in range(NUM_IMAGES):
        noise = torch.randn(1, LATENT_DIM, 1, 1).to(DEVICE)
        fake_img = generator(noise)

        save_image(
            fake_img,
            os.path.join(output_dir, f"fake_{i}.png"),
            normalize=True
        )

print(f"✅ {NUM_IMAGES} fake images generated successfully!")
