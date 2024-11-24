from torchmetrics.image import PeakSignalNoiseRatio

def calculate_psnr(preds, targets):
    psnr = PeakSignalNoiseRatio()
    return psnr(preds.detach().cpu(), targets.detach().cpu()).item()

