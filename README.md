## Deep Learning Assignment 3

### To set up the project environment, follow these steps:

### 1. Clone repository
```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
```
### 2. Run setup
```bash
    chmod +x setup.sh
    ./setup.sh
```
## How to make changes to code
### 1. Once cloned make sure working in last version, then create your working branch and checkout to that specific branch
```bash
    git pull
    git branch <branch_name>
    git checkout -b <branch_name>
```

## Part 2: Implementing a simple Convolutional classfier


### <span style="background-color:blue;">Question 7. Data preparation and training loop</span>


#### - Part (1): Data Preparation
#### Use the dataloaders to load both the training and test datasets into large tensors: one for the instances and one for the labels.\Split the training data into 50,000 training instances and 10,000 validation instances.\See implementation details in:
```bash
    src/data/load_data.py
```

#### - Part (2): Training Loop
#### Write a training loop that processes batches of 16 instances at a time.\ The training process should compute losses for each batch and validate periodically on the validation set.
```bash
    src/train/train.py
```
### <span style="background-color:blue;">Question 8. Build network and tune hyperparameters</span>
### Part(1) Build the network
```bash
    src/models/model.py
```
### Part(2) Train network
```bash
    {
    "lr": none,
    "epochs":none,
    "batch_size":none,
    "momentum":none
    }
```
### <span style="background-color:blue;">Question 9.Add data augmentation when creating the dataset for the training set</span>



## Part 3: A variable-resolution classifier
### <span style="background-color:blue;">Question 10. Data preparation and training loop</span>
```bash
    src/utils/formulas.py
    # Use this function with given parameters
    get_output_dimension():
```