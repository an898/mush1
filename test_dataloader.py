from src.dataset_utils import get_dataloaders

train_loader, val_loader, test_loader, class_names = get_dataloaders()

print("Class names:", class_names)
print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))
print("Test batches:", len(test_loader))

images, labels = next(iter(train_loader))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("First 10 labels:", labels[:10].tolist())