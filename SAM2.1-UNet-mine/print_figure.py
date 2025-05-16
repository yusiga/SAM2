import matplotlib.pyplot as plt

train_losses = []
val_losses = []

with open("loss_log.txt", "r") as f:
    lines = f.readlines()
    current_epoch = None
    last_train_loss = None

    for line in lines:
        line = line.strip()

        # 检查是否是训练 loss
        if line.startswith("epoch:"):
            parts = line.split("loss:")
            epoch_info = parts[0].strip()
            loss_value = float(parts[1].strip())
            epoch_num = int(epoch_info.split(":")[1].split("-")[0])

            # 更新当前 epoch
            if current_epoch != epoch_num:
                if last_train_loss is not None:
                    train_losses.append(last_train_loss)
                current_epoch = epoch_num

            last_train_loss = loss_value

        # 检查是否是验证 loss
        elif line.startswith("Epoch") and "Validation Loss" in line:
            val_loss = float(line.split("Validation Loss:")[1].strip())
            val_losses.append(val_loss)

    # 处理最后一个 epoch 的训练 loss
    if last_train_loss is not None:
        train_losses.append(last_train_loss)

# 绘图
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label="Train Loss", marker='o')
plt.plot(epochs, val_losses, label="Validation Loss", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()