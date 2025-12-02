import torch
import torchvision.transforms.functional as ttf
import pandas as pd
import PIL
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset



class LiTS_Segmentation_Dataset(torch.utils.data.Dataset):

    def __init__(self, csv: str, mode: str):

        self.csv = csv
        self.data = pd.read_csv(self.csv)
        self.mode = mode
        assert mode in ["train", "val", "test"] # has to be train, val, or test data - if not, assert throws an error


    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx: int):
        idx = int(idx)
        image_file = self.data.iloc[idx]["filename"]
        liver_file = self.data.iloc[idx][ "liver_segmentation"]
        tumor_file = self.data.iloc[idx][ "lesion_segmentation"]

        with PIL.Image.open(f"./Clean_LiTS/{self.mode}/{image_file}") as f:
            f = f.convert("L")
            image = ttf.pil_to_tensor(f)
        with PIL.Image.open(f"./Clean_LiTS/{self.mode}/{liver_file}") as f:
            f = f.convert("L")
            liver_mask = ttf.pil_to_tensor(f)
        with PIL.Image.open(f"./Clean_LiTS/{self.mode}/{tumor_file}") as f:
            f = f.convert("L")
            tumor_mask = ttf.pil_to_tensor(f)

        image = image.float()
        image -= image.min()
        max_val = image.max()
        if max_val > 0:
            image /= max_val


        c_targets = torch.zeros_like(liver_mask, dtype=torch.long)

        # liver = 1
        c_targets[liver_mask == 1] = 1

        # tumor = 2
        c_targets[tumor_mask == 1] = 2

        c_targets = c_targets.squeeze(0)

        oh_targets = torch.nn.functional.one_hot(c_targets, num_classes=3)
        oh_targets = oh_targets.permute(2, 0, 1).float()

        # We need both the class-index version for the cross-entropy loss and the one-hot version for our dice loss later
        return image, c_targets, oh_targets


train_dataset = LiTS_Segmentation_Dataset(csv = "./Clean_LiTS/train_classes.csv", mode="train")
val_dataset = LiTS_Segmentation_Dataset(csv = "./Clean_LiTS/val_classes.csv", mode="val")
test_dataset = LiTS_Segmentation_Dataset(csv = "./Clean_LiTS/test_classes.csv", mode="test")

batch_size = 14

# Länge des Trainingsdatensatzes
dataset_len = len(train_dataset)
num_samples = int(1 * dataset_len)  # 10%

# Zufällige Indizes auswählen
indices = torch.randperm(dataset_len)[:num_samples]

# Subset erstellen
small_train_dataset = Subset(train_dataset, indices)

train_dataloader = DataLoader(
    dataset = small_train_dataset,
    batch_size = batch_size,
    num_workers = 0,
    shuffle = True,
    drop_last = True
)

val_dataloader = DataLoader(
    dataset = val_dataset,
    batch_size = batch_size,
    num_workers = 0,
    shuffle = True,
    drop_last = True
)

test_dataloader = DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    num_workers = 0,
    shuffle = True,
    drop_last = True
)

debug_loader = DataLoader(
    dataset=train_dataset,
    batch_size=1,
    num_workers=0,    # wichtig!
    shuffle=False
)

import torch, torch.nn as nn, torch.nn.functional as nnf

def compute_dice_score(prediction: torch.Tensor, target: torch.Tensor):

    """
    Computes the dice score for one class.
    """

    prediction = prediction.to(dtype = torch.bool)
    target = target.to(dtype = torch.bool)

    intersection = torch.sum(prediction * target)   # TP
    p_cardinality = torch.sum(prediction)           # TP+FP
    t_cardinality = torch.sum(target)               # TP+FN
    cardinality = p_cardinality + t_cardinality
    eps = 1e-8

    if cardinality != 0:
        dice = (2 * intersection + eps) / (cardinality + eps) # 2*TP / (2*TP+FP+FN + eps)
    else:
        dice = None

    return dice

class BinaryDiceLoss(nn.Module):
    """
    Computes Dice loss for a single class.
    Ignores slices where both prediction and target are empty.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        # prediction: (B, H, W) probabilities
        # target: (B, H, W) one-hot
        pred_flat = prediction.reshape(prediction.size(0), -1)
        tgt_flat = target.reshape(target.size(0), -1)

        intersection = (pred_flat * tgt_flat).sum(dim=1)
        denominator = pred_flat.sum(dim=1) + tgt_flat.sum(dim=1)

        # Mask für gültige Slices (wo denominator > 0)
        valid_mask = denominator > 0
        if valid_mask.sum() == 0:
            # Wenn keine gültigen Slices, return 0
            return torch.tensor(0., device=prediction.device)

        dice = (2 * intersection[valid_mask] + self.eps) / (denominator[valid_mask] + self.eps)
        loss = 1 - dice.mean()
        return loss


class DiceLoss(nn.Module):
    """
    Computes Dice loss over all classes.
    Expects:
    - predictions: (B, C, H, W) after softmax
    - targets:     (B, C, H, W) one-hot
    """
    def __init__(self, num_classes: int = 3, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.binary_dice = BinaryDiceLoss()

        if class_weights is None:
            self.class_weights = torch.ones(num_classes)
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        losses = []
        for c in range(self.num_classes):
            pred_c = predictions[:, c, :, :]
            tgt_c = targets[:, c, :, :]
            loss_c = self.binary_dice(pred_c, tgt_c)
            loss_c = self.class_weights[c].to(predictions.device) * loss_c
            losses.append(loss_c)
        return torch.stack(losses).mean()


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, features = [64, 128, 256]):
        super(UNet, self).__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.down.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.up.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.up.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.final = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self,x):
            skip_connections = []

            for down in self.down:
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)

            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]

            for idx in range(0, len(self.up), 2):
                x = self.up[idx](x)
                skip = skip_connections[idx//2]
                concat = torch.cat([skip, x], dim=1)
                x = self.up[idx+1](concat)

            return self.final(x)

device = ("cuda" if torch.cuda.is_available() else "cpu")

model = UNet() # Your model class goes here
model = model.to(device)

dice_loss = DiceLoss(num_classes = 3, class_weights=[1.0, 5, 20]) # Your dice loss class goes here
ce_loss = nn.CrossEntropyLoss(
    weight = torch.tensor([1.0, 5, 20]).to(device = device),
    reduction = "mean",
    #ignore_index = 0
    )

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

# If your model and loss work, this should at least execute successfully.
# If you only wish to test your model, just comment out the dice_loss component everywhere.

from tqdm.auto import tqdm

num_epochs = 3
avg_liver_dice = 0
avg_lesion_dice = 0
dice_weight = 1
ce_weight = 1



for epoch in range(num_epochs):

    for step, (data, c_targets, oh_targets) in enumerate(train_dataloader):

        optimizer.zero_grad()
        data, c_targets, oh_targets = data.to(device), c_targets.to(device), oh_targets.to(device)
        predictions = model(data)

        pred_soft = torch.softmax(predictions, dim=1)
        loss_1 = dice_loss(pred_soft , oh_targets)
        loss_2 = ce_loss(predictions, c_targets)
        total_loss = loss_1 * dice_weight + loss_2 * ce_weight # We could weight contributions from the different loss components here, although 1-to-1 should do just fine

        if step % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]\t Step [{step+1}/{len(train_dataloader.dataset)//batch_size}]\t Loss: {total_loss.item():.4f}")

        total_loss.backward()
        optimizer.step()

        # Validate once after every epoch
    model.eval()

        # Don't track gradients for validation
    with torch.no_grad():

            losses = []
            background_dices = []
            background_counts = []
            liver_dices = []
            liver_counts = []
            lesion_dices = []
            lesion_counts = []
            batch_sizes = []

            for val_step, (data, c_targets, oh_targets) in enumerate(tqdm(val_dataloader)):

                data, c_targets, oh_targets = data.to(device), c_targets.to(device), oh_targets.to(device)
                predictions = model(data)
                # Choose the likeliest prediction via argmax, then convert to one-hot, and put the new axis in front again
                p_arg = nnf.one_hot(torch.argmax(predictions, dim = 1), num_classes = 3).moveaxis(-1, 1)

                # loss
                loss_1 = dice_loss(predictions, oh_targets)
                loss_2 = ce_loss(predictions, c_targets)
                total_loss = loss_1 * dice_weight + loss_2 * ce_weight # We could weight contributions from the different loss components here, although 1-to-1 should do just fine

                losses.append(total_loss.item())
                batch_sizes.append(data.size()[0])

                background_seg = oh_targets[:, 0, :, :]
                liver_seg = oh_targets[:, 1, :, :]
                lesion_seg = oh_targets[:, 2, :, :]

                background_dice = compute_dice_score(p_arg[:,0,:,:], background_seg)
                background_counts.append(data.size()[0])
                background_dices.append(background_dice)

                if liver_seg.sum() != 0.0:
                    liver_dice = compute_dice_score(p_arg[:,1,:,:], liver_seg)
                    liver_counts.append(data.size()[0])
                    liver_dices.append(liver_dice)

                if lesion_seg.sum() != 0.0:
                    lesion_dice = compute_dice_score(p_arg[:,2,:,:], lesion_seg)
                    lesion_counts.append(data.size()[0])
                    lesion_dices.append(lesion_dice)

            print(liver_dices)
            print(liver_counts)
            avg_background_dice = sum([dice * size for dice, size in zip(background_dices, background_counts)])/sum(background_counts)

            avg_liver_dice = sum([dice * size for dice, size in zip(liver_dices, liver_counts)])/sum(liver_counts)
            avg_lesion_dice = sum([dice * size for dice, size in zip(lesion_dices, lesion_counts)])/sum(lesion_counts)

            avg_loss = sum([l * bs for l, bs in zip(losses, background_counts)]) / sum(background_counts)
            print(f"Epoch: {epoch+1},\t Validation Loss: {avg_loss},\t Liver Dice Score: {avg_liver_dice}, \t Lesion Dice Score: {avg_lesion_dice}")

            # After we are done validating, let's not forget to go back to storing gradients.
            model.train()

# Test once
model.eval()

# Don't track gradients for testing
with torch.no_grad():

    losses = []
    background_dices = []
    background_counts = []
    liver_dices = []
    liver_counts = []
    lesion_dices = []
    lesion_counts = []
    batch_sizes = []

    for test_step, (data, c_targets, oh_targets) in enumerate(tqdm(test_dataloader)):

        data, c_targets, oh_targets = data.to(device), c_targets.to(device), oh_targets.to(device)
        predictions = model(data)
        # Choose the likeliest prediction via argmax, then convert to one-hot, and put the new axis in front again
        p_arg = nnf.one_hot(torch.argmax(predictions, dim = 1), num_classes = 3).moveaxis(-1, 1)

        # loss
        loss_1 = dice_loss(predictions, oh_targets)
        loss_2 = ce_loss(predictions, c_targets)
        total_loss = loss_1 * dice_weight + loss_2 * ce_weight # We could weight contributions from the different loss components here, although 1-to-1 should do just fine

        losses.append(total_loss.item())
        batch_sizes.append(data.size()[0])

        background_seg = oh_targets[:, 0, :, :]
        liver_seg = oh_targets[:, 1, :, :]
        lesion_seg = oh_targets[:, 2, :, :]

        background_dice = compute_dice_score(p_arg[:,0,:,:], background_seg)
        background_counts.append(data.size()[0])
        background_dices.append(background_dice)

        if liver_seg.sum() != 0.0:
            liver_dice = compute_dice_score(p_arg[:,1,:,:], liver_seg)
            liver_counts.append(data.size()[0])
            liver_dices.append(liver_dice)

        if lesion_seg.sum() != 0.0:
            lesion_dice = compute_dice_score(p_arg[:,2,:,:], lesion_seg)
            lesion_counts.append(data.size()[0])
            lesion_dices.append(lesion_dice)

    avg_background_dice = sum([dice * size for dice, size in zip(background_dices, background_counts)])/sum(background_counts)
    avg_liver_dice = sum([dice * size for dice, size in zip(liver_dices, liver_counts)])/sum(liver_counts)
    avg_lesion_dice = sum([dice * size for dice, size in zip(lesion_dices, lesion_counts)])/sum(lesion_counts)

    avg_loss = sum([l * bs for l, bs in zip(losses, background_counts)]) / sum(background_counts)
    print(f"Epoch: {epoch+1},\t Test Loss: {avg_loss:.4f},\t Liver Dice Score: {avg_liver_dice:.4f}, \t Lesion Dice Score: {avg_lesion_dice:.4f}")

import matplotlib.pyplot as plt
import numpy as np

def colorize_masks(gt, pred):
    """
    gt:   (H,W) ints {0,1,2}
    pred: (H,W) ints {0,1,2}

    Output: 3 RGB-Bilder
    """
    H, W = gt.shape

    gt_rgb   = np.zeros((H,W,3), dtype=np.uint8)
    pred_rgb = np.zeros((H,W,3), dtype=np.uint8)
    diff_rgb = np.zeros((H,W,3), dtype=np.uint8)

    # ========== GROUND TRUTH ==========
    gt_rgb[gt == 1] = [0, 255, 0]     # liver green
    gt_rgb[gt == 2] = [0, 150, 255]   # tumor blue

    # ========== PREDICTION ==========
    pred_rgb[pred == 1] = [255, 0, 0]   # liver red
    pred_rgb[pred == 2] = [255, 0, 255] # tumor purple

    # ========== DIFFERENCE MAP ==========
    # correct = yellow
    match = (gt == pred) & (gt != 0)
    diff_rgb[match] = [255, 255, 0]

    # false positives = red
    fp = (gt == 0) & (pred != 0)
    diff_rgb[fp] = [255, 0, 0]

    # false negatives = blue
    fn = (gt != 0) & (pred == 0)
    diff_rgb[fn] = [0, 0, 255]

    return gt_rgb, pred_rgb, diff_rgb


# ========== PLOT 9 EXAMPLES ==========
import math

model.eval()
num_examples = 100
batch_plot = 1  # wie viele Bilder gleichzeitig pro Plot (hier 1 pro Schritt)
cols = 4
rows = 10  # 10x4 = 40 Bilder pro Seite, dann mehrere Seiten

counter = 0
page = 1

with torch.no_grad():
    for data, c_targets, oh_targets in test_dataloader:

        data = data.to(device)
        preds = model(data)
        preds_arg = torch.argmax(preds, dim=1).cpu()

        batch_size = data.size(0)
        for i in range(batch_size):

            img = data.cpu()[i,0,:,:].numpy()
            gt  = c_targets[i].cpu().numpy()
            pred = preds_arg[i].numpy()

            gt_rgb, pred_rgb, diff_rgb = colorize_masks(gt, pred)

            if counter % (rows*cols) == 0:
                # neue Seite
                fig, axes = plt.subplots(rows, cols, figsize=(16, 40))
                axes = axes.flatten()

            axes[(counter % (rows*cols))*4 + 0].imshow(img, cmap="gray")
            axes[(counter % (rows*cols))*4 + 0].set_title("Input CT")
            axes[(counter % (rows*cols))*4 + 0].axis("off")

            axes[(counter % (rows*cols))*4 + 1].imshow(gt_rgb)
            axes[(counter % (rows*cols))*4 + 1].set_title("GT")
            axes[(counter % (rows*cols))*4 + 1].axis("off")

            axes[(counter % (rows*cols))*4 + 2].imshow(pred_rgb)
            axes[(counter % (rows*cols))*4 + 2].set_title("Pred")
            axes[(counter % (rows*cols))*4 + 2].axis("off")

            axes[(counter % (rows*cols))*4 + 3].imshow(diff_rgb)
            axes[(counter % (rows*cols))*4 + 3].set_title("Diff")
            axes[(counter % (rows*cols))*4 + 3].axis("off")

            counter += 1
            if counter >= num_examples:
                break

            # neue Seite anzeigen
            if counter % (rows*cols) == 0:
                plt.tight_layout()
                plt.show()
                page += 1

    # letzte Seite anzeigen, falls unvollständig
    if counter % (rows*cols) != 0:
        plt.tight_layout()
        plt.show()
