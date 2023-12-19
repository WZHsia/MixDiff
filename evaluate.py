import torch
from tqdm import tqdm
from utils import BCEDiceLoss
from utils import iou_score


def evaluate(model, dataloader, n_val, device, amp):
    model.eval()
    avg_loss, avg_dice, avg_iou, = 0, 0, 0
    val_num = len(dataloader)
    criterion = BCEDiceLoss()
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        with tqdm(total=n_val, unit='img') as pbar:
            for image, label in dataloader:
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = model(image)
                loss = criterion(pred, label)
                iou, dice = iou_score(pred, label)
                avg_loss += loss.item()
                avg_iou += iou
                avg_dice += dice
                pbar.update(image.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item(), 'iou (batch)': iou, 'dice (batch)': dice})
    return avg_loss / val_num, avg_iou / val_num, avg_dice / val_num



