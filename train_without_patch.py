import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
import time
from tqdm import tqdm


# 设置全局随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"随机种子设置为: {seed}")


set_seed(42)


class FFPPDatasetMultipleFake(Dataset):
    def __init__(self, original_root, manipulated_roots=None, mode='train',
                 seq_length=4, vectors_per_video=4, split_ratio=0.8, seed=42, verbose=True):

        self._set_seed(seed)
        self.verbose = verbose
        self.seq_length = seq_length
        self.vectors_per_video = vectors_per_video
        self.mode = mode
        self.split_ratio = split_ratio
        self.seed = seed

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: F.interpolate(x.unsqueeze(0), size=(299, 299), mode='bilinear')[0]),
                # 几何变换增强
                transforms.RandomHorizontalFlip(p=0.5),
                # 添加高斯模糊增强
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=0.3),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: F.interpolate(x.unsqueeze(0), size=(299, 299), mode='bilinear')[0]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.video_items = []
        all_samples = []

        # 加载真实样本
        if original_root and os.path.exists(original_root):
            original_samples = self._load_video_items(original_root, label=0, method='real')
            all_samples.extend(original_samples)
            print(f"加载真实样本: {len(original_samples)} 个")

        # 加载多个伪造路径的样本
        if manipulated_roots:
            method_names = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
            total_fake_samples = 0

            for i, fake_root in enumerate(manipulated_roots):
                if os.path.exists(fake_root):
                    method_name = method_names[i] if i < len(method_names) else f'fake_method_{i + 1}'
                    fake_samples = self._load_video_items(fake_root, label=1, method=method_name)
                    all_samples.extend(fake_samples)
                    total_fake_samples += len(fake_samples)
                    print(f"加载 {method_name} 伪造样本: {len(fake_samples)} 个")
                else:
                    print(f"⚠️ 伪造数据路径不存在: {fake_root}")

            print(f"总共加载伪造样本: {total_fake_samples} 个")

        # 分割数据集
        train_samples, test_samples = self._split_all_samples(all_samples)

        if mode == 'train':
            self.video_items.extend(train_samples)
        elif mode == 'test':
            self.video_items.extend(test_samples)

        # 验证数据泄露
        self._validate_no_leakage(train_samples, test_samples)

        # 计算统计信息
        real_count = sum(1 for item in self.video_items if item['label'] == 0)
        fake_count = sum(1 for item in self.video_items if item['label'] == 1)

        if self.verbose:
            print(f"{mode} set: {len(self.video_items)} videos")
            print(f"真实样本: {real_count}")
            print(f"伪造样本: {fake_count}")
            if len(self.video_items) > 0:
                print(f"真实比例: {real_count / len(self.video_items):.3f}")
                print(f"伪造比例: {fake_count / len(self.video_items):.3f}")
            else:
                print("警告: 数据集为空！")
            print("-" * 50)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _split_all_samples(self, all_samples):
        if not all_samples:
            return [], []

        video_groups = {}
        for sample in all_samples:
            video_name = sample['name']
            if video_name not in video_groups:
                video_groups[video_name] = []
            video_groups[video_name].append(sample)

        video_names = sorted(video_groups.keys())
        random.Random(self.seed).shuffle(video_names)

        split_idx = int(len(video_names) * self.split_ratio)
        train_video_names = set(video_names[:split_idx])
        test_video_names = set(video_names[split_idx:])

        train_samples = []
        test_samples = []

        for video_name, samples in video_groups.items():
            if video_name in train_video_names:
                train_samples.extend(samples)
            else:
                test_samples.extend(samples)

        print(f"确定性分割完成:")
        print(f"  总视频数: {len(video_names)}")
        print(f"  训练视频: {len(train_video_names)}")
        print(f"  测试视频: {len(test_video_names)}")

        return train_samples, test_samples

    def _validate_no_leakage(self, train_samples, test_samples):
        train_videos = set(item['name'] for item in train_samples)
        test_videos = set(item['name'] for item in test_samples)

        overlap = train_videos & test_videos

        if overlap:
            print(f"❌ 严重错误: 发现 {len(overlap)} 个重叠视频!")
            for video in list(overlap)[:5]:
                print(f"  - {video}")
            raise ValueError("数据分割失败！存在数据泄露")
        else:
            print(f"✅ 验证通过: 无数据泄露")
            print(f"  训练集唯一视频: {len(train_videos)}")
            print(f"  测试集唯一视频: {len(test_videos)}")

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        img = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def _load_video_items(self, root_path, label, method=None):
        video_items = []
        required_frames = self.seq_length * self.vectors_per_video
        skipped_videos = 0

        if not os.path.exists(root_path):
            print(f"警告：目录不存在 {root_path}")
            return video_items

        for video_name in os.listdir(root_path):
            video_path = os.path.join(root_path, video_name)
            if not os.path.isdir(video_path):
                continue

            png_files = [f for f in os.listdir(video_path) if f.endswith('.png')]

            if len(png_files) < required_frames:
                skipped_videos += 1
                continue

            selected_frames = png_files[:required_frames]

            video_items.append({
                'path': video_path,
                'label': label,
                'method': method,
                'name': video_name,
                'frames': selected_frames,
                'total_frames': len(selected_frames)
            })

        if skipped_videos > 0:
            data_type = "真实" if label == 0 else f"伪造({method})"
            print(f"跳过 {skipped_videos} 个{data_type}视频：帧数不足{required_frames}")

        return video_items

    def check_sample_differences(self, num_samples=3):
        real_samples = [item for item in self.video_items if item['label'] == 0][:num_samples]
        fake_samples = [item for item in self.video_items if item['label'] == 1][:num_samples]

        print(f"检查 {len(real_samples)} 对真实/伪造样本:")

        for i, (real, fake) in enumerate(zip(real_samples, fake_samples)):
            real_sequences, real_label, _ = self.__getitem__(self.video_items.index(real))
            fake_sequences, fake_label, _ = self.__getitem__(self.video_items.index(fake))

            print(f"\n样本对 {i}:")
            print(f"  真实: {real['name']}, 伪造: {fake['name']}")
            print(f"  数据统计:")
            print(
                f"    真实 - 范围: [{real_sequences.min():.3f}, {real_sequences.max():.3f}], 均值: {real_sequences.mean():.3f}")
            print(
                f"    伪造 - 范围: [{fake_sequences.min():.3f}, {fake_sequences.max():.3f}], 均值: {fake_sequences.mean():.3f}")
            print(f"    均值差异: {abs(real_sequences.mean() - fake_sequences.mean()):.3f}")

    def __len__(self):
        return len(self.video_items)

    def __getitem__(self, idx):
        item = self.video_items[idx]
        try:
            sequences = []
            total_frames = len(item['frames'])

            if total_frames < self.seq_length:
                raise ValueError(f"帧数不足: {total_frames} < {self.seq_length}")

            starts = np.linspace(0, total_frames - self.seq_length, num=self.vectors_per_video, dtype=int)

            for start in starts:
                frame_sequence = []
                for j in range(self.seq_length):
                    frame_path = os.path.join(item['path'], item['frames'][start + j])
                    if not os.path.exists(frame_path):
                        raise FileNotFoundError(f"文件不存在: {frame_path}")

                    frame = self._load_image(frame_path)
                    if frame is None:
                        raise ValueError(f"图像加载失败: {frame_path}")

                    frame_sequence.append(self.transform(frame))

                sequences.append(torch.stack(frame_sequence))

            sequences = torch.stack(sequences)
            label = torch.tensor(item['label'], dtype=torch.long)
            method = item['method'] if item['method'] is not None else "unknown"

            return sequences, label, method

        except Exception as e:
            print(f"加载视频 {item['name']} 出错: {str(e)}")
            dummy_sequences = torch.randn(self.vectors_per_video, self.seq_length, 3, 299, 299)
            dummy_label = torch.tensor(0, dtype=torch.long)
            dummy_method = "error"
            return dummy_sequences, dummy_label, dummy_method


class Trainer:
    def __init__(self, model, train_loader, test_loader, device, num_classes=2):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes

        # 训练配置
        self.epochs = 60
        self.lr = 8e-5

        self.best_auc = 0.0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'test_acc': [], 'test_auc': [], 'test_f1': [], 'test_recall': []
        }

    def train_complete(self):
        """完整的训练流程"""
        print("开始训练...")
        print(f"设备: {self.device}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        print(f"测试集大小: {len(self.test_loader.dataset)}")
        print(f"训练周期: {self.epochs}")
        print(f"学习率: {self.lr}")

        # 打印模型参数信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)")

        # 设置优化器和学习率调度器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch, optimizer, criterion)

            # 测试
            test_metrics = self.test_epoch()
            scheduler.step()

            # 记录历史
            self.record_history(epoch, train_loss, train_acc, test_metrics)
            self.print_epoch_results(epoch, train_loss, train_acc, test_metrics,
                                     optimizer.param_groups[0]['lr'])

            # 保存最佳模型
            if test_metrics['auc'] > self.best_auc:
                self.best_auc = test_metrics['auc']
                torch.save(self.model.state_dict(), f'best_model_auc_{self.best_auc:.4f}.pth')
                print(f"✅ 保存最佳模型，AUC: {self.best_auc:.4f}")

        print(f"\n{'=' * 60}")
        print("训练完成!")
        print(f"最佳AUC: {self.best_auc:.4f}")
        print(f"总训练周期: {self.epochs}")
        print(f"{'=' * 60}")

        return self.history

    def record_history(self, epoch, train_loss, train_acc, test_metrics):
        """记录训练历史"""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['test_acc'].append(test_metrics['accuracy'])
        self.history['test_auc'].append(test_metrics['auc'])
        self.history['test_f1'].append(test_metrics['f1'])
        self.history['test_recall'].append(test_metrics['recall'])

    def train_epoch(self, epoch, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'训练 Epoch {epoch + 1}')

        for batch_idx, (sequences, labels, methods) in enumerate(pbar):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def test_epoch(self):
        """测试一个epoch"""
        self.model.eval()
        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for sequences, labels, methods in tqdm(self.test_loader, desc='测试'):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(sequences)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_probs)
        f1 = f1_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_predictions, average='binary')

        return {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'recall': recall
        }

    def print_epoch_results(self, epoch, train_loss, train_acc, test_metrics, lr):
        """打印epoch结果"""
        print(f"\nEpoch {epoch + 1:3d}/{self.epochs} | LR: {lr:.2e}")
        print(f"训练 - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"测试 - Acc: {test_metrics['accuracy']:.4f} | AUC: {test_metrics['auc']:.4f} | "
              f"F1: {test_metrics['f1']:.4f} | Recall: {test_metrics['recall']:.4f}")
        print('-' * 80)


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # FF++数据集路径
    ffpp_real_path = "/hy-tmp/original_sequences/youtube/c23/frames"
    ffpp_fake_paths = [
        "/hy-tmp/manipulated_sequences/Deepfakes/c23/frames",
        "/hy-tmp/manipulated_sequences/Face2Face/c23/frames",
        "/hy-tmp/manipulated_sequences/FaceSwap/c23/frames",
        "/hy-tmp/manipulated_sequences/NeuralTextures/c23/frames"
    ]

    # 加载数据 - 使用所有伪造路径
    print("=== 加载FF++数据集 ===")
    print(f"真实数据路径: {ffpp_real_path}")
    print(f"伪造数据路径: {len(ffpp_fake_paths)} 个")
    for i, path in enumerate(ffpp_fake_paths, 1):
        print(f"  伪造路径 {i}: {path}")

    train_set = FFPPDatasetMultipleFake(
        original_root=ffpp_real_path,
        manipulated_roots=ffpp_fake_paths,  # 使用所有伪造路径
        mode='train',
        seed=42
    )

    test_set = FFPPDatasetMultipleFake(
        original_root=ffpp_real_path,
        manipulated_roots=ffpp_fake_paths,  # 使用所有伪造路径
        mode='test',
        seed=42
    )

    train_set.check_sample_differences(3)

    train_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    # 创建模型 - 使用新的XceptionTemporal1D模型
    from model_without_patch import XceptionTemporal1D

    # 指定预训练权重路径
    pretrained_path = "/hy-tmp/xception-43020ad28.pth"

    model = XceptionTemporal1D(
        num_classes=2,
        seq_length=4,
        pretrained_path=pretrained_path
    )
    model = model.to(device)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device
    )

    # 开始训练
    history = trainer.train_complete()

    # 保存训练历史
    torch.save(history, 'ffpp_training_history.pth')
    print("训练历史已保存")


if __name__ == "__main__":
    main()