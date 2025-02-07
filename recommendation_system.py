import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import optuna
from typing import List, Dict, Optional, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict
from datetime import datetime
import random

# Converts input to a PyTorch tensor on the given device, optionally with a specified dtype
def safe_tensor(x, device, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if dtype is not None:
            x = x.to(dtype)
        return x.to(device)
    return torch.tensor(x, dtype=dtype, device=device)

# Context manager to measure elapsed time
class Profiler:
    def __init__(self):
        self.start_time = None
        self.end_time = None
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
    @property
    def elapsed(self) -> float:
        return (self.end_time - self.start_time).total_seconds() if self.end_time else 0.0

# Hybrid model combining matrix factorization and content-based features
class HybridRecommender(nn.Module):
    def __init__(self, num_users: int, num_items: int, latent_dim: int, content_dim: int = 100):
        super(HybridRecommender, self).__init__()
        # Embedding layers for users and items
        self.user_embeddings = nn.Embedding(num_users, latent_dim)
        self.item_embeddings = nn.Embedding(num_items, latent_dim)
        # MLP to process content features combined with user embeddings
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + content_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(latent_dim, 1)
        )
        # Mean Squared Error loss
        self.loss_fn = nn.MSELoss()
        # Adam optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, content_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Lookup embeddings
        ue = self.user_embeddings(user_ids.long())
        ie = self.item_embeddings(item_ids.long())
        # Matrix factorization output
        mf_output = torch.sum(ue * ie, dim=1, keepdim=True)
        # If content features are provided, merge them with user embeddings and feed to MLP
        if content_features is not None:
            x = torch.cat([ue, content_features], dim=1)
            return (mf_output + self.mlp(x)) / 2.0
        return mf_output
    # Single training step
    def train_step(self, user_ids: torch.Tensor, item_ids: torch.Tensor, ratings: torch.Tensor, content_features: Optional[torch.Tensor] = None) -> float:
        self.optimizer.zero_grad()
        preds = self.forward(user_ids, item_ids, content_features)
        loss = self.loss_fn(preds.squeeze(), ratings.float())
        loss.backward()
        self.optimizer.step()
        return loss.item()

# Dataset to hold user, item, rating, and optional content features
class RecDataset(Dataset):
    def __init__(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray, content_features: Optional[np.ndarray] = None):
        self.users = users
        self.items = items
        self.ratings = ratings
        self.content_features = content_features
    def __len__(self) -> int:
        return len(self.users)
    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        u = self.users[idx]
        i = self.items[idx]
        r = self.ratings[idx]
        # Return content features if they exist
        if self.content_features is not None:
            return (u, i, r, self.content_features[idx])
        return (u, i, r)

# Generates content-based features using TF-IDF followed by scaling
class ContentFeatureGenerator:
    def __init__(self):
        self.scaler = StandardScaler()
    def generate_content_features(self, item_descriptions: np.ndarray) -> np.ndarray:
        tfidf_vectorizer = TfidfVectorizer(max_features=100)
        features = tfidf_vectorizer.fit_transform(item_descriptions).toarray()
        return self.scaler.fit_transform(features)

# Manages training and evaluation of the HybridRecommender model
class ModelTrainer:
    def __init__(self, model: HybridRecommender, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(torch.float32)
        if self.device != "cpu" and not torch.cuda.is_available():
            self.device = "cpu"
        self.model.to(self.device)
    def train(self, train_loader: DataLoader, valid_loader: DataLoader, epochs: int = 10) -> Dict:
        history = defaultdict(list)
        for _ in range(epochs):
            train_loss = []
            # Train loop
            with Profiler():
                for batch in train_loader:
                    if len(batch) == 4:
                        u, i, r, cf = batch
                        u = safe_tensor(u, self.device, torch.float32)
                        i = safe_tensor(i, self.device, torch.float32)
                        r = safe_tensor(r, self.device, torch.float32)
                        cf = safe_tensor(cf, self.device, torch.float32)
                        loss = self.model.train_step(u, i, r, cf)
                    else:
                        u, i, r = batch
                        u = safe_tensor(u, self.device, torch.float32)
                        i = safe_tensor(i, self.device, torch.float32)
                        r = safe_tensor(r, self.device, torch.float32)
                        loss = self.model.train_step(u, i, r)
                    train_loss.append(loss)
                history["train_loss"].append(np.mean(train_loss))
            # Validation loop
            val_loss = []
            with torch.no_grad():
                for batch in valid_loader:
                    if len(batch) == 4:
                        u, i, r, cf = batch
                        u = safe_tensor(u, self.device, torch.float32)
                        i = safe_tensor(i, self.device, torch.float32)
                        r = safe_tensor(r, self.device, torch.float32)
                        cf = safe_tensor(cf, self.device, torch.float32)
                        preds = self.model(u, i, cf)
                        val_loss.append(self.model.loss_fn(preds.squeeze(), r).item())
                    else:
                        u, i, r = batch
                        u = safe_tensor(u, self.device, torch.float32)
                        i = safe_tensor(i, self.device, torch.float32)
                        r = safe_tensor(r, self.device, torch.float32)
                        preds = self.model(u, i)
                        val_loss.append(self.model.loss_fn(preds.squeeze(), r).item())
                history["val_loss"].append(np.mean(val_loss))
        return history
    # Computes RMSE and MAE on a given dataset
    def evaluate(self, data_loader: DataLoader) -> Dict:
        self.model.eval()
        predictions, true_ratings = [], []
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 4:
                    u, i, r, cf = batch
                    u = safe_tensor(u, self.device, torch.float32)
                    i = safe_tensor(i, self.device, torch.float32)
                    r = safe_tensor(r, self.device, torch.float32)
                    cf = safe_tensor(cf, self.device, torch.float32)
                    preds = self.model(u, i, cf).cpu().numpy()
                else:
                    u, i, r = batch
                    u = safe_tensor(u, self.device, torch.float32)
                    i = safe_tensor(i, self.device, torch.float32)
                    r = safe_tensor(r, self.device, torch.float32)
                    preds = self.model(u, i).cpu().numpy()
                predictions.extend(preds.squeeze())
                true_ratings.extend(r.cpu().numpy())
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(true_ratings))**2))
        mae = np.mean(np.abs(np.array(predictions) - np.array(true_ratings)))
        return {"RMSE": rmse, "MAE": mae}

# Objective function for Optuna hyperparameter optimization
def optimize_hyperparameters(trial: optuna.Trial, train_loader: DataLoader, valid_loader: DataLoader, num_users: int, num_items: int, content_dim: int) -> float:
    ld = trial.suggest_int("latent_dim", 32, 128)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dr = trial.suggest_float("dropout_rate", 0.0, 0.5)
    # Create and configure a new HybridRecommender with trial parameters
    model = HybridRecommender(num_users, num_items, ld, content_dim)
    for layer in model.mlp:
        if isinstance(layer, nn.Dropout):
            layer.p = dr
    model.optimizer.param_groups[0]["lr"] = lr
    trainer = ModelTrainer(model)
    # Train for a few epochs to estimate validation loss
    history = trainer.train(train_loader, valid_loader, epochs=3)
    return history["val_loss"][-1]

# Generates synthetic rating data for a given number of users and items
class SyntheticDataGenerator:
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
    def generate_ratings(self):
        data = []
        # Each user gets 10 random interactions
        for _ in range(self.num_users * 10):
            u = random.randint(0, self.num_users - 1)
            i = random.randint(0, self.num_items - 1)
            r = random.uniform(0, 5)
            data.append([u, i, r])
        return np.array(data)

# Main function to orchestrate data generation, model creation, training, and evaluation
def main() -> None:
    num_users = 1000
    num_items = 500
    content_dim = 100
    # Create synthetic ratings data
    data_generator = SyntheticDataGenerator(num_users, num_items)
    ratings_matrix = data_generator.generate_ratings()
    # Split data into train, validation, and test
    all_indices = np.arange(len(ratings_matrix))
    train_indices, val_test_indices = train_test_split(all_indices, test_size=0.3, random_state=42)
    val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=42)
    # Generate content-based features
    content_generator = ContentFeatureGenerator()
    item_descriptions = np.array([f"Item {i} description." for i in range(num_items)])
    content_features = content_generator.generate_content_features(item_descriptions)
    # Create datasets
    train_dataset = RecDataset(
        ratings_matrix[train_indices, 0],
        ratings_matrix[train_indices, 1],
        ratings_matrix[train_indices, 2],
        content_features[ratings_matrix[train_indices, 1].astype(int)]
    )
    val_dataset = RecDataset(
        ratings_matrix[val_indices, 0],
        ratings_matrix[val_indices, 1],
        ratings_matrix[val_indices, 2],
        content_features[ratings_matrix[val_indices, 1].astype(int)]
    )
    test_dataset = RecDataset(
        ratings_matrix[test_indices, 0],
        ratings_matrix[test_indices, 1],
        ratings_matrix[test_indices, 2],
        content_features[ratings_matrix[test_indices, 1].astype(int)]
    )
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # Use Optuna to find the best hyperparameters
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: optimize_hyperparameters(t, train_loader, val_loader, num_users, num_items, content_dim),
        n_trials=5,
        timeout=300
    )
    # Retrieve the best parameters and configure a final model
    bp = study.best_trial.params
    final_model = HybridRecommender(num_users, num_items, bp["latent_dim"], content_dim)
    for layer in final_model.mlp:
        if isinstance(layer, nn.Dropout):
            layer.p = bp["dropout_rate"]
    final_model.optimizer.param_groups[0]["lr"] = bp["learning_rate"]
    # Train the final model
    final_trainer = ModelTrainer(final_model)
    final_trainer.train(train_loader, val_loader, epochs=5)
    # Evaluate the final model on the test set
    metrics = final_trainer.evaluate(test_loader)
    print("Final Metrics:", metrics)

if __name__ == "__main__":
    main()
