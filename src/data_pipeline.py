from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_data():
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,),(0.3081,))
  ])

  train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
  test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

  train_size = int(0.9 * len(train_ds))
  val_size = len(train_ds) - train_size
  train_set, val_set = random_split(train_ds, [train_size, val_size])

  train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
  val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
  test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

  return train_loader, val_loader, test_loader