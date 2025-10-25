import torch
import torch.nn as nn
import torch.optim as optim
from data import config_path, random_seed, model_checkpoint_path
from cifar10_exp_preprocessing import cifar10_preprocessing, get_resnet18_cifar10
from utils import read_config
import time


def main():
    print("Reading configuration...", config_path)
    config = read_config(config_path)
    data_params = config["experiment"]["data_params"]
    train_params = config["experiment"]["train_params"]
    val_split, batch_size = data_params["val_split"], data_params["batch_size"]
    epochs, lr = train_params["epochs"], train_params["lr"]

    train_loader, val_loader, test_loader = cifar10_preprocessing(val_split=val_split,batch_size=batch_size, seed=random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    

    model = get_resnet18_cifar10().to(device)

    # --- Optimizer, Scheduler, Loss ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    start_time = time.time()
    # --- Training Loop ---
    def train_one_epoch(epoch):
        model.train()
        total, correct, total_loss = 0, 0, 0.0
        for step,(x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

        acc = correct / total
        avg_loss = total_loss / total
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, train_acc={acc*100:.2f}%")
        return avg_loss, acc


    def evaluate(loader, name="val"):
        model.eval()
        total, correct, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                total_loss += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)

        acc = correct / total
        avg_loss = total_loss / total
        print(f"{name}_loss={avg_loss:.4f}, {name}_acc={acc*100:.2f}%")
        return avg_loss, acc

    # --- Train ---
    for epoch in range(1, epochs + 1):
        train_one_epoch(epoch)
        evaluate(val_loader, "val")
        scheduler.step()
    print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes.")
    
    # --- Final Test ---
    print("Evaluating on test set...")
    evaluate(test_loader, "test")

    #---- Savig the model ----
    torch.save({
    'epoch': epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'scaler_state': scaler.state_dict() if 'scaler' in locals() else None,
    }, model_checkpoint_path)



if __name__ == "__main__":
    main()
