# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Regression is a supervised learning task where the goal is to predict a continuous numerical value based on input features. Neural networks, specifically deep learning models, are used for regression tasks to capture complex patterns and relationships in data. The objective is to develop a neural network regression model that learns from a given dataset and makes accurate predictions based on unseen inputs.

## Neural Network Model

![image](https://github.com/user-attachments/assets/d84ed18c-30fb-46fc-988a-5f96b73361f0)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Krithick Vivekananda
### Register Number: 212223240075
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,8)
        self.fc3=nn.Linear(8,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x


# Initialize the Model, Loss Function, and Optimizer
ai_brain= NeuralNet()
criterion= nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train,criterion,optimizer,epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(ai_brain(X_train),y_train)
    loss.backward()
    optimizer.step()

    ai_brain.history['loss'].append(loss.item())
    if epoch % 200==0:
      print(f'Epoch [{epoch}/{epochs}],Loss:{loss.item():.6f}')

```
## Dataset Information

![image](https://github.com/user-attachments/assets/53048b6c-75bf-4408-8293-500368d35fff)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/c5556a59-8674-48c7-813f-c9f84dda61ad)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/d65ef9b8-eb8f-4ad4-8607-af1faa7e61ad)

## RESULT

The neural network regression model was successfully developed, trained, and tested. The performance of the model was analyzed using training loss plots and sample data predictions, confirming that the model effectively learned patterns from the dataset.
