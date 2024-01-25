import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from datetime import datetime
import os
import torch
import numpy as np
import logging
from datetime import datetime
import os
from typing import Literal

# # 配置logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.addHandler(logging.FileHandler("training.log"))  # 将日志信息写入文件


def train(
    model,
    train_loader,
    val_loader,
    num_epochs=200,
    patience=20,
    learning_rate=0.001,
    lr_decay_epochs=5,  # Number of epochs before decaying the learning rate
    device="cpu",
    model_save_name="best_model.pth",  # Default model save filename
    input_feature: Literal["fhr", "ucp", "both"] = "fhr",
):
    # Use current time to generate a unique log filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = model_save_name.split(".")[0]
    log_filename = f"{log_name}_{current_time}.log"
    model_save_path = os.path.join("./checkpoint", model_save_name)

    # Configure logging to output logs to both console and file
    log_filepath = os.path.join("./log", log_filename)
    logging.basicConfig(level=logging.INFO, filename=log_filepath)
    console_handler = logging.StreamHandler()
    logger = logging.getLogger(__name__)
    logger.addHandler(console_handler)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_model_state_dict = model.state_dict()
    patience_count = 0
    decay_count = 0

    # 将model的形状存入到log中
    logger.info(model)

    for epoch in range(num_epochs):
        model.train()

        _train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            logger,
            epoch,
            num_epochs,
            input_feature=input_feature,
        )

        model.eval()

        val_accuracy, val_labels, val_losses = _evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        mean_val_accuracy = np.mean(val_accuracy)
        mean_val_loss = np.mean(val_losses)

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] - Validation Accuracy: {mean_val_accuracy:.4f} - Validation Loss: {mean_val_loss:.4f}"
        )

        if mean_val_accuracy > best_val_accuracy:
            best_val_loss = mean_val_loss
            best_val_accuracy = mean_val_accuracy
            best_model_state_dict = model.state_dict()
            patience_count = 0
            _save_model(model, model_save_path, logger)
        else:
            patience_count += 1
            logger.info(f"Early stopping patience count: {patience_count}/{patience}")

        if patience_count >= patience:
            logger.info(f"Early stopping after {patience} epochs of no improvement.")
            logger.info(
                f"Best Validation Accuracy: {best_val_accuracy:.4f} - Best Validation Loss: {best_val_loss:.4f}"
            )
            break

        # Decay learning rate every lr_decay_epochs epochs using cosine annealing
        if (epoch + 1) % lr_decay_epochs == 0:
            learning_rate = (
                learning_rate * 0.5 * (1 + np.cos((epoch + 1) / num_epochs * np.pi))
            )
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            decay_count += 1
            logger.info(
                f"Learning rate decayed ({decay_count} times). New learning rate: {learning_rate}"
            )

    model.load_state_dict(best_model_state_dict)
    return model


def _train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    logger,
    epoch,
    num_epochs,
    input_feature: Literal["fhr", "ucp", "both"] = "fhr",
):
    for batch_idx, (fhr, ucp, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        fhr = fhr.unsqueeze(1).to(device)
        ucp = ucp.unsqueeze(1).to(device)
        combined = torch.cat((fhr, ucp), dim=1)

        if input_feature == "fhr":
            inputs = fhr
            inputs = inputs.permute(0, 2, 1)
        elif input_feature == "ucp":
            inputs = ucp
            inputs = inputs.permute(0, 2, 1)
        elif input_feature == "both":
            combined = torch.cat((fhr, ucp), dim=1)
            inputs = combined
            inputs = inputs.permute(0, 2, 1)
        else:
            raise ValueError(
                "Invalid input_feature value. Use 'fhr', 'ucp', or 'both'."
            )

        labels = labels.to(device)
        enc_mark = torch.ones(8, 4800).to(device)
        outputs = model(inputs, enc_mark)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] - Batch [{batch_idx+1}/{len(train_loader)}] - Training Loss: {loss.item():.4f}"
            )


def _evaluate(
    model,
    val_loader,
    criterion,
    device,
    input_feature: Literal["fhr", "ucp", "both"] = "fhr",
):
    val_accuracy = []
    val_labels = []
    val_losses = []

    with torch.no_grad():
        for fhr, ucp, labels in val_loader:
            fhr = fhr.unsqueeze(1).to(device)
            ucp = ucp.unsqueeze(1).to(device)
            combined = torch.cat((fhr, ucp), dim=1)

            if input_feature == "fhr":
                inputs = fhr
                inputs = inputs.permute(0, 2, 1)
            elif input_feature == "ucp":
                inputs = ucp
                inputs = inputs.permute(0, 2, 1)
            elif input_feature == "both":
                combined = torch.cat((fhr, ucp), dim=1)
                inputs = combined
                inputs = inputs.permute(0, 2, 1)
            else:
                raise ValueError(
                    "Invalid input_feature value. Use 'fhr', 'ucp', or 'both'."
                )

            labels = labels.to(device)
            enc_mark = torch.ones(1, 4800).to(device)
            outputs = model(inputs, enc_mark)

            _, predicted = torch.max(outputs, 1)
            val_accuracy.extend((predicted == labels).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            val_loss = criterion(outputs, labels)
            val_losses.append(val_loss.item())

    return val_accuracy, val_labels, val_losses


def _save_model(model, model_save_path, logger):
    torch.save(model.state_dict(), model_save_path)
    logger.info(
        f"----------Saving model with the best accuracy to {model_save_path}.----------"
    )


# 模型评估的代码
def eval(
    model,
    val_loader,
    device,
    logger=logging.getLogger(__name__),
    input_feature: Literal["fhr", "ucp", "both"] = "fhr",
):
    # 将模型移动到指定设备
    model = model.to(device)
    model.eval()

    # label为0表示正常，label为1表示异常
    # 正确率，即预测正确的样本数占总样本数的比例
    # 精确率，即预测为正样本的样本中，真正为正样本的比例
    # 召回率，即真正为正样本的样本中，预测为正样本的比例
    # F1-score，即精确率和召回率的调和平均数
    # TP：预测为正样本，实际为正样本
    # FP：预测为正样本，实际为负样本
    # TN：预测为负样本，实际为负样本
    # FN：预测为负样本，实际为正样本

    accuracy = []
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for fhr, ucp, labels in val_loader:
        # 将输入数据移动到指定设备
        fhr = fhr.unsqueeze(1).to(device)
        ucp = ucp.unsqueeze(1).to(device)
        combined = torch.cat((fhr, ucp), dim=1)
        labels = labels.to(device)

        if input_feature == "fhr":
            inputs = fhr
            inputs = inputs.permute(0, 2, 1)
        elif input_feature == "ucp":
            inputs = ucp
            inputs = inputs.permute(0, 2, 1)
        elif input_feature == "both":
            combined = torch.cat((fhr, ucp), dim=1)
            inputs = combined
            inputs = inputs.permute(0, 2, 1)
        else:
            raise ValueError(
                "Invalid input_feature value. Use 'fhr', 'ucp', or 'both'."
            )

        with torch.no_grad():
            enc_mark = torch.ones(1, 4800).to(device)
            outputs = model(inputs, enc_mark)

        _, predicted = torch.max(outputs, 1)
        accuracy.extend((predicted == labels).cpu().numpy())
        # val_labels.extend(labels.cpu().numpy())

        # 计算TP，FP，TN，FN
        TP += ((predicted == labels) & (predicted == 1)).cpu().numpy().sum()
        FP += ((predicted != labels) & (predicted == 1)).cpu().numpy().sum()
        TN += ((predicted == labels) & (predicted == 0)).cpu().numpy().sum()
        FN += ((predicted != labels) & (predicted == 0)).cpu().numpy().sum()

    accuracy = np.mean(accuracy)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)

    # 记录在log中
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1-score: {f1_score:.4f}")
