You now have multiple users, each with their own sequence of actions (e.g., ```[0, 1, 5, 1, 7, 8, ...]```), and you want to predict the next action for each user individually.

## 🧠 Approaches

### ✅ **Train One Global Model (Shared across all users)**

- All users’ data is used together.
- The model learns general patterns in sequences.

> 👉 This is the most common and practical approach.

Example:
Input  →  Output
[2, 3, 4] → [3] or [3,2]
[3, 4, 3] → [6] or [5,1]
[0, 1, 6] → [7] or [5,2]

---

### ✅ **Train a Separate Model per User**

- Build a model for each individual user.
- Useful if:
  - Users have very different behaviors.
  - You have **a lot of data per user**.
  - You want **personalized predictions**.

> 👉 More complex, but more personalized.

#### 🧠 2. Can We Train Thousands of Models?
> 🧠 **Yes**, you can create a model per user even for **thousands of users**, especially if the models are small (like lightweight LSTMs or Transformers).  


### ✅ **One Model with User Embeddings**

Instead of training thousands of models, consider using **one global model** that includes **user identity as an embedding**.

#### Given data:

```python
all_users = {
    'user_1': [0, 1, 5, 1, 7, 8],
    'user_2': [2, 3, 4, 3, 6],
    'user_3': [0, 1, 6, 7, 8],
}
```
---

## 🧱 Step 1: Map Usernames to IDs

### 🟩 Input:

```python
all_users = {
    'user_1': [...],
    'user_2': [...],
    ...
}
```

### ⚙️ Processing:

We map each username to a numeric user ID for use in embeddings.

```python
user_to_id = {'user_1': 0, 'user_2': 1, 'user_3': 2}
```

### 🟦 Output:

```python
user_1 → 0
user_2 → 1
user_3 → 2
```

---

## 🧱 Step 2: Create Input-Output Pairs from Sequences (For Supervised Learning)

### ⚙️ Processing:

We generate `(input_sequence, user_id, target_action)`:

| Input Sequence | User ID | Target |
| -------------- | ------- | ------ |
| `[0, 1, 5]`    | `0`     | `1`    |
| `[1, 5, 1]`    | `0`     | `7`    |
| `[5, 1, 7]`    | `0`     | `8`    |

### 🟦 Output: Training Samples (For Supervised Learning)

```python
samples = [
    ([0, 1, 5], 0, 1),
    ([1, 5, 1], 0, 7),
    ([5, 1, 7], 0, 8),
    ([2, 3, 4], 1, 3),
    ([3, 4, 3], 1, 6),
    ([0, 1, 6], 2, 7),
    ([1, 6, 7], 2, 8),
]
```

---

## 🧱 Step 3: Convert Inputs to Tensors for the Model

### 🟩 Input:

One training sample: `([0, 1, 5], 0, 1)`

### ⚙️ Processing:

Use `torch.tensor()` to convert each:

```python
input_seq = torch.tensor([0, 1, 5])    # shape: (3,)
user_id   = torch.tensor(0)            # shape: (1,)
target    = torch.tensor(1)            # shape: (1,)
```

Batch them in DataLoader later → shape becomes:

```python
input_seqs: (batch_size, 3)
user_ids:   (batch_size,)
targets:    (batch_size,)
```

---

## 🧱 Step 4: Model Forward Pass

### 🟩 Input to model:

* `input_seqs = [[0, 1, 5], [1, 5, 1], ...]`  (batch of sequences)
* `user_ids = [0, 0, ...]`

### ⚙️ Processing:

1. **Action Embedding:**
   Each action ID is turned into a dense vector → shape becomes `(batch_size, seq_len, embed_dim)`

2. **LSTM Layer:**
   Outputs a hidden state summarizing the sequence → shape: `(batch_size, hidden_dim)`

3. **User Embedding:**
   Turns user IDs into vectors → shape: `(batch_size, user_embed_dim)`

4. **Concatenation:**
   Combine LSTM output and user embedding → shape: `(batch_size, hidden_dim + user_embed_dim)`

5. **Linear Layer (Classifier):**
   Predicts a probability distribution over all actions → shape: `(batch_size, num_actions)`

### 🟦 Output:

Predicted next action (as logits or probabilities). Example:

```python
[0.1, 0.05, 0.8, 0.03, 0.02] → predicted = 2 (highest prob)
```

---

## 🧱 Step 5: Prediction for a Specific User

### 🟩 Input:

User sequence: `[5, 1, 7]`, user = `"user_1"`

### ⚙️ Processing:

1. Look up user ID:

   ```python
   user_id = user_to_id['user_1']  # → 0
   ```

2. Convert sequence and user to tensor:

   ```python
   input_seq = torch.tensor([5, 1, 7]).unsqueeze(0)  # shape: (1, 3)
   user_id = torch.tensor([0])                      # shape: (1,)
   ```

3. Run model:

   ```python
   logits = model(input_seq, user_id)
   ```

4. Get prediction:

   ```python
   predicted_action = torch.argmax(logits, dim=1).item()
   ```

### 🟦 Output:

Model returns the **most likely next action** for `user_1` after `[5, 1, 7]`.

---

## Summary Table

| Step               | Input                   | Output                        |
| ------------------ | ----------------------- | ----------------------------- |
| User → ID Mapping  | `'user_1'`              | `0`                           |
| Sequence → Samples | `[0,1,5,1,7]`           | `([0,1,5], 0, 1)` and so on   |
| Convert to tensors | `([0,1,5], 0, 1)`       | PyTorch tensors               |
| Model forward      | input\_seq + user\_id   | action probabilities (logits) |
| Prediction         | `[0.1, 0.8, 0.05, ...]` | `1` (predicted next action)   |

---


## Code

### Overview

| Component                     | Role                                                                 |
| ----------------------------- | -------------------------------------------------------------------- |
| `ActionSequenceDataset`       | Prepares sequences of actions + user ID + target action              |
| `PersonalizedActionPredictor` | Model using LSTM + user embedding to predict next action             |
| `train_model()`               | Trains the model                                                     |
| `predict_next_action()`       | Predicts the next action for a given user and action sequence        |
| `main()`                      | Sets up everything, loads data, trains, and runs a sample prediction |


### Python

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple

# === Constants ===
SEQ_LEN = 3
BATCH_SIZE = 2
EMBED_DIM = 32
USER_EMBED_DIM = 16
HIDDEN_DIM = 64
EPOCHS = 10
LEARNING_RATE = 0.001


# === Dataset Class ===
class ActionSequenceDataset(Dataset):
    def __init__(self, user_data: Dict[str, List[int]], seq_len: int):
        self.seq_len = seq_len
        self.user_to_id = {user: idx for idx, user in enumerate(user_data)}
        self.id_to_user = {idx: user for user, idx in self.user_to_id.items()}
        self.samples = self._generate_samples(user_data)

    def _generate_samples(self, user_data: Dict[str, List[int]]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        samples = []
        for user, actions in user_data.items():
            uid = self.user_to_id[user]
            for i in range(len(actions) - self.seq_len):
                seq = torch.tensor(actions[i:i + self.seq_len], dtype=torch.long)
                target = torch.tensor(actions[i + self.seq_len], dtype=torch.long)
                user_tensor = torch.tensor(uid, dtype=torch.long)
                samples.append((seq, user_tensor, target))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# === Model Class ===
class PersonalizedActionPredictor(nn.Module):
    def __init__(self, num_actions: int, num_users: int):
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, EMBED_DIM)
        self.user_embedding = nn.Embedding(num_users, USER_EMBED_DIM)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM + USER_EMBED_DIM, num_actions)

    def forward(self, action_seq: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        embedded_actions = self.action_embedding(action_seq)  # (batch, seq_len, embed_dim)
        _, (hidden, _) = self.lstm(embedded_actions)          # hidden: (1, batch, hidden_dim)
        hidden = hidden.squeeze(0)                            # (batch, hidden_dim)
        user_embed = self.user_embedding(user_ids)            # (batch, user_embed_dim)
        combined = torch.cat((hidden, user_embed), dim=1)     # (batch, hidden_dim + user_embed_dim)
        return self.fc(combined)


# === Training Function ===
def train_model(model: nn.Module, dataloader: DataLoader, criterion, optimizer, epochs: int):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for action_seqs, user_ids, targets in dataloader:
            optimizer.zero_grad()
            predictions = model(action_seqs, user_ids)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


# === Prediction Function ===
def predict_next_action(model: nn.Module, user_sequence: List[int], user_name: str, dataset: ActionSequenceDataset) -> int:
    model.eval()
    with torch.no_grad():
        user_id = torch.tensor([dataset.user_to_id[user_name]])
        input_seq = torch.tensor(user_sequence[-SEQ_LEN:]).unsqueeze(0)
        logits = model(input_seq, user_id)
        return torch.argmax(logits, dim=1).item()


# === Main Logic ===
def main():
    all_users = {
        'user_1': [0, 1, 5, 1, 7, 8],
        'user_2': [2, 3, 4, 3, 6],
        'user_3': [0, 1, 6, 7, 8],
    }

    # Compute vocab sizes
    unique_actions = set(a for seq in all_users.values() for a in seq)
    num_actions = max(unique_actions) + 1
    num_users = len(all_users)

    # Dataset & Dataloader
    dataset = ActionSequenceDataset(all_users, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model, Loss, Optimizer
    model = PersonalizedActionPredictor(num_actions=num_actions, num_users=num_users)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    train_model(model, dataloader, criterion, optimizer, EPOCHS)

    # Predict for a user
    test_seq = [5, 1, 7]
    prediction = predict_next_action(model, test_seq, 'user_1', dataset)
    print(f"Predicted next action for user_1 after {test_seq}: {prediction}")


# === Run it ===
if __name__ == "__main__":
    main()
```
