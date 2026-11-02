import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import *
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
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
    """支持多个真实路径和多个伪造路径的数据集类"""

    def __init__(self, original_roots=None, manipulated_roots=None, mode='test',
                 seq_length=4, vectors_per_video=4, split_ratio=0.8, seed=42, verbose=True):

        self._set_seed(seed)
        self.verbose = verbose
        self.seq_length = seq_length
        self.vectors_per_video = vectors_per_video
        self.mode = mode
        self.split_ratio = split_ratio
        self.seed = seed

        # 测试时只做必要的预处理，不做增强
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: F.interpolate(x.unsqueeze(0), size=(299, 299), mode='bilinear')[0]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.video_items = []

        # 加载所有样本（多个真实路径+多个伪造路径）
        all_samples = []

        # 加载多个真实路径的样本
        if original_roots:
            method_names = ['real_celeb', 'real_youtube', 'real_actors']  # 可以根据需要修改
            total_real_samples = 0

            for i, real_root in enumerate(original_roots):
                if os.path.exists(real_root):
                    method_name = method_names[i] if i < len(method_names) else f'real_method_{i + 1}'
                    real_samples = self._load_video_items(real_root, label=0, method=method_name)
                    all_samples.extend(real_samples)
                    total_real_samples += len(real_samples)
                    print(f"加载 {method_name} 真实样本: {len(real_samples)} 个")
                else:
                    print(f"⚠️ 真实数据路径不存在: {real_root}")

            print(f"总共加载真实样本: {total_real_samples} 个")
        else:
            print("⚠️ 没有提供真实数据路径")

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
        else:
            print("⚠️ 没有提供伪造数据路径")

        # 对于测试模式，使用所有样本
        self.video_items.extend(all_samples)

        # 计算统计信息
        real_count = sum(1 for item in self.video_items if item['label'] == 0)
        fake_count = sum(1 for item in self.video_items if item['label'] == 1)

        # 按方法详细统计
        method_stats = {}
        for item in self.video_items:
            method = item['method']
            if method not in method_stats:
                method_stats[method] = {'real': 0, 'fake': 0}
            if item['label'] == 0:
                method_stats[method]['real'] += 1
            else:
                method_stats[method]['fake'] += 1

        if self.verbose:
            print(f"{mode} set: {len(self.video_items)} videos")
            print(f"真实样本: {real_count}")
            print(f"伪造样本: {fake_count}")
            if len(self.video_items) > 0:
                print(f"真实比例: {real_count / len(self.video_items):.3f}")
                print(f"伪造比例: {fake_count / len(self.video_items):.3f}")
            else:
                print("警告: 数据集为空！")

            # 打印详细的方法统计
            print("\n详细方法统计:")
            for method, counts in method_stats.items():
                total = counts['real'] + counts['fake']
                label_type = "真实" if counts['real'] > 0 else "伪造"
                print(f"  {method}: {total} 个样本 ({label_type})")
            print("-" * 50)

    def check_sample_differences(self, num_samples=3):
        """检查数据集中真实和伪造样本的差异"""
        real_samples = [item for item in self.video_items if item['label'] == 0][:num_samples]
        fake_samples = [item for item in self.video_items if item['label'] == 1][:num_samples]

        print(f"检查 {len(real_samples)} 对真实/伪造样本:")

        for i, (real, fake) in enumerate(zip(real_samples, fake_samples)):
            real_sequences, real_label, _ = self.__getitem__(self.video_items.index(real))
            fake_sequences, fake_label, _ = self.__getitem__(self.video_items.index(fake))

            print(f"\n样本对 {i}:")
            print(f"  真实: {real['name']} ({real['method']}), 伪造: {fake['name']} ({fake['method']})")
            print(f"  数据统计:")
            print(
                f"    真实 - 范围: [{real_sequences.min():.3f}, {real_sequences.max():.3f}], 均值: {real_sequences.mean():.3f}")
            print(
                f"    伪造 - 范围: [{fake_sequences.min():.3f}, {fake_sequences.max():.3f}], 均值: {fake_sequences.mean():.3f}")
            print(f"    均值差异: {abs(real_sequences.mean() - fake_sequences.mean()):.3f}")

    def _set_seed(self, seed):
        """设置确定性随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _split_all_samples(self, all_samples):
        """一次性分割所有样本，确保训练集和测试集完全分离"""
        if not all_samples:
            return [], []

        # 按完整视频名称分组
        video_groups = {}
        for sample in all_samples:
            video_name = sample['name']
            if video_name not in video_groups:
                video_groups[video_name] = []
            video_groups[video_name].append(sample)

        # 确定性分割
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
        """严格验证无数据泄露"""
        train_videos = set(item['name'] for item in train_samples)
        test_videos = set(item['name'] for item in test_samples)

        overlap = train_videos & test_videos

        if overlap:
            print(f"❌ 严重错误: 发现 {len(overlap)} 个重叠视频!")
            print("重叠视频示例:")
            for video in list(overlap)[:5]:
                print(f"  - {video}")
            raise ValueError("数据分割失败！存在数据泄露")
        else:
            print(f"✅ 验证通过: 无数据泄露")
            print(f"  训练集唯一视频: {len(train_videos)}")
            print(f"  测试集唯一视频: {len(test_videos)}")

    def _load_image(self, path):
        """使用Pillow加载图像并转换为tensor"""
        try:
            img = Image.open(path).convert('RGB')
            img = np.array(img).astype(np.float32) / 255.0
            return torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
        except Exception as e:
            print(f"加载图像失败 {path}: {e}")
            return None

    def _create_black_image(self, height=320, width=320):
        """创建全黑图像"""
        black_img = np.zeros((height, width, 3), dtype=np.float32)
        return torch.from_numpy(black_img).permute(2, 0, 1)  # [C, H, W]

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

            # 获取所有PNG文件
            png_files = [f for f in os.listdir(video_path) if f.endswith('.png')]

            # 不再跳过帧数不足的视频，而是允许加载
            if len(png_files) < required_frames:
                print(f"视频 {video_name} 帧数不足: {len(png_files)} < {required_frames}，将用黑帧补全")

            # 使用所有可用的帧
            selected_frames = png_files[:min(len(png_files), required_frames)]

            video_items.append({
                'path': video_path,
                'label': label,
                'method': method,
                'name': video_name,
                'frames': selected_frames,
                'total_frames': len(selected_frames)
            })

        return video_items

    def __len__(self):
        return len(self.video_items)

    def __getitem__(self, idx):
        item = self.video_items[idx]
        try:
            sequences = []
            total_frames = len(item['frames'])
            required_frames = self.seq_length * self.vectors_per_video

            # 如果帧数不足，用黑帧补全
            if total_frames < required_frames:
                # 创建黑帧
                black_frame = self._create_black_image()
                black_frame = self.transform(black_frame)

                # 复制现有帧并用黑帧补全
                available_frames = []
                for frame_name in item['frames']:
                    frame_path = os.path.join(item['path'], frame_name)
                    if os.path.exists(frame_path):
                        frame = self._load_image(frame_path)
                        if frame is not None:
                            transformed_frame = self.transform(frame)
                            available_frames.append(transformed_frame)

                # 用黑帧补全到required_frames
                while len(available_frames) < required_frames:
                    available_frames.append(black_frame.clone())

                # 更新总帧数
                total_frames = len(available_frames)
                frames_to_use = available_frames
            else:
                # 帧数足够，正常加载
                frames_to_use = []
                for frame_name in item['frames'][:required_frames]:
                    frame_path = os.path.join(item['path'], frame_name)
                    if not os.path.exists(frame_path):
                        raise FileNotFoundError(f"文件不存在: {frame_path}")

                    frame = self._load_image(frame_path)
                    if frame is None:
                        raise ValueError(f"图像加载失败: {frame_path}")

                    transformed_frame = self.transform(frame)
                    frames_to_use.append(transformed_frame)

            # 生成起始位置
            starts = np.linspace(0, total_frames - self.seq_length, num=self.vectors_per_video, dtype=int)

            for start in starts:
                frame_sequence = []
                for j in range(self.seq_length):
                    frame_sequence.append(frames_to_use[start + j])

                sequences.append(torch.stack(frame_sequence))

            sequences = torch.stack(sequences)  # [vectors, seq_len, C, H, W]
            label = torch.tensor(item['label'], dtype=torch.long)
            method = item['method'] if item['method'] is not None else "unknown"

            return sequences, label, method

        except Exception as e:
            print(f"加载视频 {item['name']} 出错: {str(e)}")
            # 返回有效的备用数据，确保不是None
            dummy_sequences = torch.randn(self.vectors_per_video, self.seq_length, 3, 299, 299)
            dummy_label = torch.tensor(0, dtype=torch.long)
            dummy_method = "error"
            return dummy_sequences, dummy_label, dummy_method


class ModelTester:
    def __init__(self, model, test_loader, device, model_path):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model_path = model_path

        # 加载模型权重
        self._load_model()

    def _load_model(self):
        """加载训练好的模型权重"""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')

            if 'state_dict' in checkpoint:
                # 完整检查点格式
                state_dict = checkpoint['state_dict']
            else:
                # 直接state_dict格式
                state_dict = checkpoint

            # 加载权重
            self.model.load_state_dict(state_dict)
            print(f"✅ 成功加载模型权重: {self.model_path}")

            # 如果有最佳AUC信息，打印出来
            if 'best_auc' in checkpoint:
                print(f"模型训练时的最佳AUC: {checkpoint['best_auc']:.4f}")

        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            raise

    def test(self):
        """在测试集上评估模型"""
        print(f"\n{'=' * 60}")
        print("开始模型测试")
        print(f"{'=' * 60}")

        self.model.eval()
        all_labels = []
        all_predictions = []
        all_probs = []
        all_methods = []

        inference_times = []

        with torch.no_grad():
            for sequences, labels, methods in tqdm(self.test_loader, desc='测试进度'):
                sequences = sequences.to(self.device)

                # 计算推理时间
                start_time = time.time()
                outputs = self.model(sequences)
                end_time = time.time()
                inference_times.append((end_time - start_time) / sequences.size(0))  # 每个样本的平均时间

                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_methods.extend(methods)

        # 计算各项指标
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probs, inference_times, all_methods)

        # 打印详细结果
        self._print_detailed_results(metrics, all_labels, all_predictions, all_probs, all_methods)

        return metrics

    def _calculate_metrics(self, all_labels, all_predictions, all_probs, inference_times, all_methods):
        """计算评估指标"""
        accuracy = accuracy_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_probs)
        f1 = f1_score(all_labels, all_predictions, average='binary')
        recall = recall_score(all_labels, all_predictions, average='binary')
        precision = precision_score(all_labels, all_predictions, average='binary')

        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        tn, fp, fn, tp = cm.ravel()

        # 计算推理时间统计
        avg_inference_time = np.mean(inference_times) * 1000  # 转换为毫秒
        std_inference_time = np.std(inference_times) * 1000

        # 按方法统计
        method_results = {}
        unique_methods = set(all_methods)
        for method in unique_methods:
            method_indices = [i for i, m in enumerate(all_methods) if m == method]
            if len(method_indices) > 0:
                method_labels = [all_labels[i] for i in method_indices]
                method_preds = [all_predictions[i] for i in method_indices]
                method_probs = [all_probs[i] for i in method_indices]

                method_acc = accuracy_score(method_labels, method_preds)
                method_f1 = f1_score(method_labels, method_preds, average='binary')
                method_precision = precision_score(method_labels, method_preds, average='binary')
                method_recall = recall_score(method_labels, method_preds, average='binary')

                method_results[method] = {
                    'samples': len(method_indices),
                    'accuracy': method_acc,
                    'f1': method_f1,
                    'precision': method_precision,
                    'recall': method_recall
                }

        return {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'confusion_matrix': cm,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'true_positive': tp,
            'avg_inference_time_ms': avg_inference_time,
            'std_inference_time_ms': std_inference_time,
            'total_samples': len(all_labels),
            'method_results': method_results
        }

    def _print_detailed_results(self, metrics, all_labels, all_predictions, all_probs, all_methods):
        """打印详细测试结果"""
        print(f"\n{'=' * 80}")
        print("模型测试结果")
        print(f"{'=' * 80}")

        print(f"总样本数: {metrics['total_samples']}")
        print(f"真实样本: {sum(all_labels)}")
        print(f"伪造样本: {len(all_labels) - sum(all_labels)}")
        print()

        print("主要指标:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC:      {metrics['auc']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  Precision:{metrics['precision']:.4f}")
        print(f"  Recall:   {metrics['recall']:.4f}")
        print()

        print("混淆矩阵:")
        print(f"            Predicted")
        print(f"            Real    Fake")
        print(f"Actual Real  {metrics['true_negative']:6d}  {metrics['false_positive']:6d}")
        print(f"       Fake  {metrics['false_negative']:6d}  {metrics['true_positive']:6d}")
        print()

        print("性能指标:")
        print(
            f"  平均推理时间: {metrics['avg_inference_time_ms']:.2f} ± {metrics['std_inference_time_ms']:.2f} ms/样本")
        print(f"  吞吐量: {1000 / metrics['avg_inference_time_ms']:.2f} 样本/秒")
        print()

        # 按方法统计
        if metrics['method_results']:
            print("按方法统计:")
            print(f"{'方法':<15} {'样本数':<8} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10}")
            print("-" * 80)
            for method_name, method_metrics in metrics['method_results'].items():
                print(
                    f"{method_name:<15} {method_metrics['samples']:<8} {method_metrics['accuracy']:.4f}    {method_metrics['f1']:.4f}    {method_metrics['precision']:.4f}    {method_metrics['recall']:.4f}")
            print()

        print(f"{'=' * 80}")


def test_on_multiple_datasets(original_roots, manipulated_roots, model_path, batch_size=8):
    """在多个真实路径和多个伪造路径上测试模型"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 检查路径是否存在
    valid_original_roots = []
    if original_roots:
        for original_root in original_roots:
            if os.path.exists(original_root):
                valid_original_roots.append(original_root)
            else:
                print(f"⚠️ 真实数据路径不存在，跳过: {original_root}")

    valid_manipulated_roots = []
    if manipulated_roots:
        for manipulated_root in manipulated_roots:
            if os.path.exists(manipulated_root):
                valid_manipulated_roots.append(manipulated_root)
            else:
                print(f"⚠️ 伪造数据路径不存在，跳过: {manipulated_root}")

    if not valid_original_roots and not valid_manipulated_roots:
        print("❌ 没有有效的真实或伪造数据路径")
        return None

    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return None

    # 加载测试数据（合并多个真实路径和多个伪造路径）
    print("加载测试数据集...")
    test_set = FFPPDatasetMultipleFake(
        original_roots=valid_original_roots,
        manipulated_roots=valid_manipulated_roots,
        mode='test',
        seed=42
    )

    if len(test_set) == 0:
        print("❌ 测试集为空，请检查数据路径")
        return None

    test_set.check_sample_differences(2)

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 创建模型
    model = XceptionTemporal1DWithPatches(num_classes=2, seq_length=4)
    model = model.to(device)

    # 创建测试器并运行测试
    tester = ModelTester(model, test_loader, device, model_path)
    results = tester.test()

    return results


def main():
    """主函数 - 支持多个真实路径和多个伪造数据路径"""

    # 指定多个真实路径和伪造路径
    original_roots = [
        "/hy-tmp/celeb-df-v1/Celeb-real/frames",
        "/hy-tmp/celeb-df-v1/YouTube-real/frames"  # 可以添加更多真实路径
    ]  # 多个真实数据路径

    manipulated_roots = [
        "/hy-tmp/celeb-df-v1/Celeb-synthesis/frames"
    ]  # 多个伪造数据路径

    model_path = "/hy-tmp/best_model_phase2_auc_0.9703_trained_on_ff++_second_time.pth"
    batch_size = 16

    print("测试配置:")
    print(f"真实数据路径: {len(original_roots)} 个")
    for i, path in enumerate(original_roots, 1):
        print(f"  真实路径 {i}: {path}")
    print(f"伪造数据路径: {len(manipulated_roots)} 个")
    for i, path in enumerate(manipulated_roots, 1):
        print(f"  伪造路径 {i}: {path}")
    print(f"模型权重路径: {model_path}")
    print(f"批次大小: {batch_size}")
    print("-" * 50)

    # 运行测试
    try:
        results = test_on_multiple_datasets(
            original_roots=original_roots,
            manipulated_roots=manipulated_roots,
            model_path=model_path,
            batch_size=batch_size
        )

        # 保存结果到文件
        if results:
            result_file = f"test_results_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write("Deepfake检测模型测试结果\n")
                f.write("=" * 50 + "\n")
                f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"真实数据路径数量: {len(original_roots)}\n")
                for i, path in enumerate(original_roots, 1):
                    f.write(f"  真实路径 {i}: {path}\n")
                f.write(f"伪造数据路径数量: {len(manipulated_roots)}\n")
                for i, path in enumerate(manipulated_roots, 1):
                    f.write(f"  伪造路径 {i}: {path}\n")
                f.write(f"模型路径: {model_path}\n")
                f.write(f"总样本数: {results['total_samples']}\n\n")

                f.write("总体性能指标:\n")
                f.write(f"Accuracy:  {results['accuracy']:.4f}\n")
                f.write(f"AUC:       {results['auc']:.4f}\n")
                f.write(f"F1-Score:  {results['f1']:.4f}\n")
                f.write(f"Precision: {results['precision']:.4f}\n")
                f.write(f"Recall:    {results['recall']:.4f}\n")
                f.write(
                    f"推理时间:  {results['avg_inference_time_ms']:.2f} ± {results['std_inference_time_ms']:.2f} ms/样本\n\n")

                # 按方法统计
                if 'method_results' in results:
                    f.write("按方法统计:\n")
                    f.write(
                        f"{'方法':<15} {'样本数':<8} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10}\n")
                    f.write("-" * 80 + "\n")
                    for method_name, method_metrics in results['method_results'].items():
                        f.write(
                            f"{method_name:<15} {method_metrics['samples']:<8} {method_metrics['accuracy']:.4f}    {method_metrics['f1']:.4f}    {method_metrics['precision']:.4f}    {method_metrics['recall']:.4f}\n")

            print(f"\n✅ 测试结果已保存到: {result_file}")

    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()