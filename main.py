import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef

import matplotlib.pyplot as plt

from libauc.losses import AUCMLoss

from preprocessor import MolecularPreprocessor

from model_2d import GINEncoder

from model_3d import GeoDirNet

from contrast import ContrastiveLearningModule

from classifier import ClassificationHead

from wrapper import MultiTaskWrapper

def train_epoch(model, loader, optimizer, criterion_cls, lambda_contrast, lambda_auc, auc_module, device):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss_contrast = out['contrast_loss']
        logits = out['cls_logits']
        labels = data.y.float()

        loss_cls = criterion_cls(logits, labels)

        probs = torch.sigmoid(logits)
        loss_auc = auc_module(probs, labels)

        loss = loss_cls + lambda_contrast * loss_contrast + lambda_auc * loss_auc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

        probs_np = probs.detach().cpu().numpy()
        preds_np = (probs_np > 0.5).astype(int)
        labels_np = labels.cpu().numpy()

        all_labels.extend(labels_np)
        all_probs.extend(probs_np)
        all_preds.extend(preds_np)

    train_acc = accuracy_score(all_labels, all_preds)
    train_auc = roc_auc_score(all_labels, all_probs)
    train_f1 = f1_score(all_labels, all_preds)
    train_mcc = matthews_corrcoef(all_labels, all_preds)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, train_acc, train_auc, train_f1, train_mcc

def eval_epoch(model, loader, criterion_cls, lambda_contrast, device):
    model.eval()
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss_contrast = out['contrast_loss']
            logits = out['cls_logits']
            labels = data.y.float()

            loss_cls = criterion_cls(logits, labels)
            loss = loss_cls + lambda_contrast * loss_contrast
            total_loss += loss.item() * data.num_graphs

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels_np = labels.cpu().numpy()

            all_labels.extend(labels_np)
            all_probs.extend(probs)
            all_preds.extend(preds)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    mcc = matthews_corrcoef(all_labels, all_preds)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, acc, auc, f1, mcc

class MoleculeDataset(Dataset):
    def __init__(self, data_dir: str, preprocessor: MolecularPreprocessor, split: str = 'train'):
        self.cache_path = os.path.join(data_dir, f"{split}_cached.pt")
        if os.path.exists(self.cache_path):
            print(f"[CACHE] Loading cached dataset from {self.cache_path}")
            self.data_list = torch.load(self.cache_path)
            return

        print(f"[PREPROCESS] Processing {split}.csv, this might take a while...")
        path = os.path.join(data_dir, f"{split}.csv")
        df = pd.read_csv(path)
        self.data_list = []

        for i, (smi, label) in enumerate(zip(df['SMILES'], df['Label'])):
            result = preprocessor.process_smiles(smi)
            if result is None:
                continue
            data_2d, coords_3d = result
            data_2d.pos = torch.tensor(coords_3d, dtype=torch.float)
            data_2d.y = torch.tensor(label, dtype=torch.long)
            self.data_list.append(data_2d)
            if i % 100 == 0:
                print(f"Processed {i}/{len(df)} molecules")

        if len(self.data_list) == 0:
            raise RuntimeError(f"No valid molecules in {split}.csv!")

        # 保存缓存
        torch.save(self.data_list, self.cache_path)
        print(f"[CACHE] Saved cached dataset to {self.cache_path}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def plot_metric_trends(epochs, accs, aucs, f1s, mccs, save_path=None):

    plt.figure(figsize=(10, 8))

    plt.plot(epochs, accs, label='Accuracy', marker='o')
    plt.plot(epochs, aucs, label='AUC', marker='s')
    plt.plot(epochs, f1s, label='F1 Score', marker='^')
    plt.plot(epochs, mccs, label='MCC', marker='d')

    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Metric Trends Over Epochs')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def generate_predictions_csv(model, data_dir, preprocessor, device, fold_id):

    model.eval()

    test_csv_path = os.path.join(data_dir, 'test.csv')
    test_df = pd.read_csv(test_csv_path)

    results = []

    with torch.no_grad():
        for idx, row in test_df.iterrows():
            smiles = row['SMILES']
            true_label = row['Label']

            result = preprocessor.process_smiles(smiles)
            if result is None:
                results.append({
                    'SMILES': smiles,
                    'True_Label': int(true_label),
                    'Predicted_Label': 0,
                    'Predicted_Probability': 0.5
                })
                continue

            data_2d, coords_3d = result
            data_2d.pos = torch.tensor(coords_3d, dtype=torch.float)
            data_2d.y = torch.tensor(true_label, dtype=torch.long)

            data_2d.batch = torch.zeros(data_2d.x.shape[0], dtype=torch.long)

            data_2d = data_2d.to(device)

            out = model(data_2d)
            logits = out['cls_logits']
            prob = torch.sigmoid(logits).cpu().item()
            pred = int(prob > 0.5)

            results.append({
                'SMILES': smiles,
                'True_Label': int(true_label),
                'Predicted_Label': pred,
                'Predicted_Probability': prob
            })

    results_df = pd.DataFrame(results)

    csv_filename = os.path.join(data_dir, f'test_predictions_fold_{fold_id}.csv')
    results_df.to_csv(csv_filename, index=False)

    print(f"\n✓ 预测结果已保存到: {csv_filename}")
    print(f"  - 总样本数: {len(results_df)}")
    print(f"  - 预测为不稳定: {(results_df['Predicted_Label'] == 1).sum()}")
    print(f"  - 预测为稳定: {(results_df['Predicted_Label'] == 0).sum()}")


    return results_df

def run_once(fold_id=0, seed=None):
    import random
    import torch
    import numpy as np

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    data_dir = f'10fold_data/fold_{fold_id}'
    batch_size = 128
    lr = 1e-5
    epochs = 80
    lambda_contrast = 0.1
    lambda_auc = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    preprocessor = MolecularPreprocessor()
    train_set = MoleculeDataset(data_dir, preprocessor, split='train')
    val_set = MoleculeDataset(data_dir, preprocessor, split='test')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    encoder2d = GINEncoder(133, 256, 256).to(device)
    encoder3d = GeoDirNet(133, 256, 128).to(device)
    contrast = ContrastiveLearningModule(512, 128, 64, 0.1).to(device)
    cls_head = ClassificationHead(512, 128, 192, 96, 0.1).to(device)
    model = MultiTaskWrapper(encoder2d, encoder3d, contrast, cls_head).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion_cls = nn.BCEWithLogitsLoss()
    auc_module = AUCMLoss()

    best_auc = 0.0
    best_sum = 0.0
    best_metrics = {}

    print("开始训练！")

    epochs_list = []
    acc_list = []
    auc_list = []
    f1_list = []
    mcc_list = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_auc, train_f1, train_mcc = train_epoch(
            model, train_loader, optimizer,
            criterion_cls, lambda_contrast,
            lambda_auc, auc_module, device
        )
        val_loss, val_acc, val_auc, val_f1, val_mcc = eval_epoch(
            model, val_loader, criterion_cls,
            lambda_contrast, device
        )

        epochs_list.append(epoch)
        acc_list.append(val_acc)
        auc_list.append(val_auc)
        f1_list.append(val_f1)
        mcc_list.append(val_mcc)

        print((f"Epoch {epoch:03d} | "
               f"TrainLoss: {train_loss:.4f} | "
               f"Acc: {train_acc:.4f} | AUC: {train_auc:.4f} | "
               f"F1: {train_f1:.4f} | MCC: {train_mcc:.4f} || "
               f"ValLoss: {val_loss:.4f} | "
               f"Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | "
               f"F1: {val_f1:.4f} | MCC: {val_mcc:.4f}"), end='')

        if val_auc > best_auc:
            best_auc = val_auc
            best_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_f1": val_f1,
                "val_mcc": val_mcc,
            }
            model_save_path = f'{data_dir}/best_model_auc.pt'
            torch.save(model.state_dict(), model_save_path)
            print("  <-- Best so far!", flush=True)
        else:
            print()

    print("\n🏆 Best Model Based on AUC:")
    print((f"Epoch {best_metrics['epoch']:03d} | "
           f"TrainLoss: {best_metrics['train_loss']:.4f} | "
           f"ValLoss: {best_metrics['val_loss']:.4f} | "
           f"Acc: {best_metrics['val_acc']:.4f} | "
           f"AUC: {best_metrics['val_auc']:.4f} | "
           f"F1: {best_metrics['val_f1']:.4f} | "
           f"MCC: {best_metrics['val_mcc']:.4f}"))

    plot_metric_trends(
        epochs_list,
        acc_list,
        auc_list,
        f1_list,
        mcc_list,
        save_path=f'{data_dir}/metric_trends.png'
    )

    print("\n生成预测CSV文件...")
    best_model_path = f'{data_dir}/best_model_auc.pt'
    model.load_state_dict(torch.load(best_model_path))

    generate_predictions_csv(
        model=model,
        data_dir=data_dir,
        preprocessor=preprocessor,
        device=device,
        fold_id=fold_id
    )

    return best_metrics


def main():

    fold_id = 2
    print(f"\n==================== Running Fold {fold_id} ====================")

    best_metrics = run_once(fold_id=fold_id, seed=43)

    print("\n📊 Training Complete! Final Results:")
    print(f"Best Epoch:   {best_metrics['epoch']}")
    print(f"Val Loss:     {best_metrics['val_loss']:.4f}")
    print(f"Val Accuracy: {best_metrics['val_acc']:.4f}")
    print(f"Val AUC:      {best_metrics['val_auc']:.4f}")
    print(f"Val F1:       {best_metrics['val_f1']:.4f}")
    print(f"Val MCC:      {best_metrics['val_mcc']:.4f}")


if __name__ == '__main__':
    main()