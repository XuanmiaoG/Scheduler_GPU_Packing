import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time
import os
import csv
import re
from torchsummary import summary
from ptflops import get_model_complexity_info

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def load_cifar10_data(data_folder, model_folder):
    if model_folder == 'vit_b_16':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])

    test_set = datasets.CIFAR10(root=data_folder, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader

def test_model_inference_and_accuracy(model, device, data_loader):
    correct = 0
    total = 0
    total_inference_time = 0
    single_image_times = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            
            inference_time = (end_time - start_time)
            total_inference_time += inference_time
            single_image_times.extend([inference_time / images.size(0)] * images.size(0))
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_single_image_time = total_inference_time / total
    
    return total_inference_time, accuracy, avg_single_image_time



def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb

def get_macs_and_params(model, input_size):
    macs, params = get_model_complexity_info(model, input_size, as_strings=False,
                                             print_per_layer_stat=False, verbose=False)
    return macs, params

def write_result_to_csv(model_folder, variant, inference_time, accuracy, model_size, macs, avg_single_image_time, models_dir):
    filename = os.path.join(models_dir, model_folder, f"{model_folder}_inference_results.csv")
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['Model', 'Variant', 'Pruning Amount', 'Inference Time (s)', 'Avg Single Image Time (s)', 'Accuracy (%)', 'Model Size (MB)', 'MACs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        # Extract pruning amount or check if it's the original model
        pruning_match = re.search(r'pruned_(\d+\.\d+)', variant)
        pruning_amount = pruning_match.group(1) if pruning_match else 'original'
        
        writer.writerow({
            'Model': model_folder, 
            'Variant': variant, 
            'Pruning Amount': pruning_amount, 
            'Inference Time (s)': f"{inference_time:.3f}", 
            'Avg Single Image Time (s)': f"{avg_single_image_time:.6f}",
            'Accuracy (%)': accuracy,
            'Model Size (MB)': f"{model_size:.3f}",
            'MACs': macs
        })


def scan_models(models_dir):
    model_details = {}
    for folder in os.listdir(models_dir):
        folder_path = os.path.join(models_dir, folder)
        if os.path.isdir(folder_path):
            model_details[folder] = [file for file in os.listdir(folder_path) if file.endswith('.pth')]
    return model_details

def main(models_dir, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_details = scan_models(models_dir)

    for folder, files in model_details.items():
        test_data_loader = load_cifar10_data(data_dir, folder)
        for file_name in files:
            model_path = os.path.join(models_dir, folder, file_name)
            model = load_model(model_path, device)
            total_inference_time, accuracy, avg_single_image_time = test_model_inference_and_accuracy(
                model, device, test_data_loader
            )
            model_size = get_model_size(model)
            macs, _ = get_macs_and_params(model, (3, 224, 224) if folder == 'vit_b_16' else (3, 32, 32))
            write_result_to_csv(
                folder, file_name, total_inference_time, accuracy, model_size, macs, avg_single_image_time, models_dir
            )
            print(f"Avg single image inference time: {avg_single_image_time:.6f} seconds")


if __name__ == "__main__":
    models_directory = "./models"
    data_directory = "./data"
    main(models_directory, data_directory)
