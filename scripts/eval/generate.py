import os
import torch
import torchvision.utils as vutils
from models import Generator
import config

# Set device
device = torch.device("cuda:0" if (torch.cuda.is_available() and config.ngpu > 0) else "cpu")

# Load Generator
netG = Generator(config.ngpu, config.nz, config.ngf, config.nc).to(device)
weights_path = "checkpoints/netG_epoch_30.pth"  # Change this to desired epoch
netG.load_state_dict(torch.load(weights_path, map_location=device))
netG.eval()

# Generate noise
num_images = 64
noise = torch.randn(num_images, config.nz, 1, 1, device=device)

# Generate fake images
with torch.no_grad():
    fake_images = netG(noise).detach().cpu()

# Save image grid
os.makedirs("generated", exist_ok=True)
vutils.save_image(fake_images, "generated/generated_sample.png", normalize=True, nrow=8)

print("✅ Generated images saved to: generated/generated_sample.png")
