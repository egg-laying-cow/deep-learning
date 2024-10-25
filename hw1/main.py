import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, lr=0.001):
        super(MLPClassifier, self).__init__()

        # Lớp đầu vào đến lớp đầu ra
        self.layer = nn.Linear(input_dim, output_dim)
        
        # Loss function và optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.layer(x)
        return x

    def fit(self, dataloader, epochs=20):
        self.train()
        for epoch in range(epochs):
            for inputs, labels in dataloader:
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            _, predictions = torch.max(outputs, 1)
        return predictions

    def evaluate(self, dataloader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f'Accuracy: {accuracy * 100:.2f}%')
        return accuracy

def main():
    # Tải bộ dữ liệu Iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # Chuyển đổi dữ liệu thành Tensor và tạo DataLoader
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Khởi tạo mô hình với input_dim=4, output_dim=3 (số lớp phân loại trong Iris)
    input_dim = X_train.shape[1]
    output_dim = 3
    model = MLPClassifier(input_dim, output_dim)

    # Huấn luyện mô hình
    model.fit(train_loader, epochs=200)

    # Đánh giá mô hình trên tập kiểm tra
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model.evaluate(test_loader)

    predictions = model.predict(X_test_tensor[:5])
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
