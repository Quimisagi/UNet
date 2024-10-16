import torch
from model import UNet
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from torch.utils.data import random_split
from utils import CarvanaDataset
import random

from utils import dice_coefficient

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    num_workers = torch.cuda.device_count() * 4
else:
    num_workers = 4


DATASET_DIR = '/home/quimisagi/Daigaku/Practice/Datasets/Carvana'
generator = torch.Generator().manual_seed(25)

BATCH_SIZE = 8

train_dataset = CarvanaDataset(DATASET_DIR)

train_dataset, test_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)
test_dataloader = DataLoader(dataset=test_dataset,
                            num_workers=num_workers, pin_memory=False,
                            batch_size=BATCH_SIZE,
                            shuffle=True)


model_pth = './my_checkpoint.pth'
trained_model = UNet(in_channels=3, num_classes=1).to(device)
trained_model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

criterion = nn.BCEWithLogitsLoss()

test_running_loss = 0
test_running_dc = 0

# with torch.no_grad():
#     for idx, img_mask in enumerate(tqdm(test_dataloader, position=0, leave=True)):
#         img = img_mask[0].float().to(device)
#         mask = img_mask[1].float().to(device)

#         y_pred = trained_model(img)
#         loss = criterion(y_pred, mask)
#         dc = dice_coefficient(y_pred, mask)

#         test_running_loss += loss.item()
#         test_running_dc += dc.item()

#     test_loss = test_running_loss / (idx + 1)
#     test_dc = test_running_dc / (idx + 1)


# print(f"Test Loss: {test_loss:.4f}, Test Dice Coefficient: {test_dc:.4f}")

def random_images_inference(image_tensors, mask_tensors, image_paths, model_pth, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize((512, 512))
    ])

    # Iterate for the images, masks and paths
    for image_pth, mask_pth, image_paths in zip(image_tensors, mask_tensors, image_paths):
        # Load the image
        img = transform(image_pth)
        
        # Predict the imagen with the model
        pred_mask = model(img.unsqueeze(0))
        pred_mask = pred_mask.squeeze(0).permute(1,2,0)
        
        # Load the mask to compare
        mask = transform(mask_pth).permute(1, 2, 0).to(device)
        
        print(f"Image: {os.path.basename(image_paths)}, DICE coefficient: {round(float(dice_coefficient(pred_mask, mask)),5)}")
        
        # Show the images
        img = img.cpu().detach().permute(1, 2, 0)
        pred_mask = pred_mask.cpu().detach()
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 0] = 1
        
        plt.figure(figsize=(15, 16))
        plt.subplot(131), plt.imshow(img), plt.title("original")
        plt.subplot(132), plt.imshow(pred_mask, cmap="gray"), plt.title("predicted")
        plt.subplot(133), plt.imshow(mask, cmap="gray"), plt.title("mask")
        plt.show()

n = 10

image_tensors = []
mask_tensors = []
image_paths = []

for _ in range(n):
    random_index = random.randint(0, len(test_dataloader.dataset) - 1)
    random_sample = test_dataloader.dataset[random_index]

    image_tensors.append(random_sample[0])  
    mask_tensors.append(random_sample[1]) 
    image_paths.append(random_sample[2])

random_images_inference(image_tensors, mask_tensors, image_paths, model_pth, device="cpu")
