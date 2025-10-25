
import torch
import torch.optim as optim

import numpy as np
from utils import read_config
import torch.nn.functional as F
from data import config_path, model_checkpoint_path, random_seed
from cifar10_exp_preprocessing import cifar10_preprocessing, get_resnet18_cifar10
import matplotlib.pyplot as plt


def main():
    print("Reading configuration...", config_path)
    config = read_config(config_path)
    data_params = config["experiment"]["data_params"]
    train_params = config["experiment"]["train_params"]
    val_split, batch_size = data_params["val_split"], data_params["batch_size"]
    epochs, lr = train_params["epochs"], train_params["lr"]

    _, val_loader, test_loader = cifar10_preprocessing(val_split=val_split,batch_size=batch_size, seed=random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet18_cifar10().to(device)


    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    if checkpoint['scaler_state'] is not None:
        scaler.load_state_dict(checkpoint['scaler_state'])
    
    start_epoch = checkpoint['epoch'] + 1
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            correct += model(x).argmax(1).eq(y).sum().item()
            total += y.size(0)
    print(f"Val acc: {100*correct/total:.2f}%")

    #### ECE ####
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = F.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    correct = preds.eq(labels)


    def expected_calibration_error(confs, correct, n_bins=15):
        confs = confs.numpy()
        correct = correct.numpy()
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            bin_lower, bin_upper = bins[i], bins[i + 1]
            in_bin = (confs > bin_lower) & (confs <= bin_upper)
            prop = in_bin.mean()
            if prop > 0:
                acc = correct[in_bin].mean()
                avg_conf = confs[in_bin].mean()
                ece += np.abs(acc - avg_conf) * prop
        return ece

    ece = expected_calibration_error(confs, correct)
    print(f"ECE: {ece:.4f}")


    def reliability_plot(confs, correct, n_bins=15):
        confs = confs.numpy()
        correct = correct.numpy()
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        accs, avg_confs = [], []
        for i in range(n_bins):
            bin_lower, bin_upper = bins[i], bins[i + 1]
            in_bin = (confs > bin_lower) & (confs <= bin_upper)
            if in_bin.any():
                accs.append(correct[in_bin].mean())
                avg_confs.append(confs[in_bin].mean())
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(avg_confs, accs, marker='o')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Reliability Diagram')
        plt.show()

    reliability_plot(confs, correct)



if __name__ == "__main__":
    main()
