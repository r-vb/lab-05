# TicTacToe

import random

# Initialize the board
def initialize_board():
    return [' ' for _ in range(9)]  # List of 9 spaces representing the Tic Tac Toe grid

# Display the board
def print_board(board):
    for i in range(0, 9, 3):
        print(f"{board[i]} | {board[i+1]} | {board[i+2]}")
        if i < 6:
            print("--+---+--")

# Check if a player has won
def check_win(board, player):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    for condition in win_conditions:
        if all(board[i] == player for i in condition):
            return True
    return False

# Check if the board is full
def is_board_full(board):
    return ' ' not in board

# Get the available moves
def available_moves(board):
    return [i for i in range(9) if board[i] == ' ']

# AI's move function - now uses random moves
def ai_move(board):
    return random.choice(available_moves(board))  # AI randomly chooses an available move

# Function to handle the player's move
def player_move(board, player_symbol):
    while True:
        try:
            move = int(input(f"Enter your move ({player_symbol}): ")) - 1
            if move not in available_moves(board):
                print("Invalid move. Try again.")
            else:
                board[move] = player_symbol
                break
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 9.")

# Main game loop
def main():
    board = initialize_board()
    print("Welcome to Tic Tac Toe!")
    
    # Let the user choose whether they want 'X' or 'O'
    player_choice = ''
    while player_choice not in ['X', 'O']:
        player_choice = input("Do you want to play as 'X' or 'O'? ").upper()
    
    # AI plays the opposite symbol of the player
    ai_choice = 'O' if player_choice == 'X' else 'X'
    
    # Let the user choose whether they want to go first or not
    first_move = ''
    while first_move not in ['yes', 'no']:
        first_move = input("Do you want to go first? (yes/no) ").lower()

    # Determine who starts based on player choice and first move preference
    if player_choice == 'O' and first_move == 'no':
        player_turn = False  # AI goes first if player chose 'O' and said no
    elif player_choice == 'X' and first_move == 'no':
        player_turn = False  # AI goes first if player chose 'X' and said no
    else:
        player_turn = True  # Player goes first if they chose 'O' and said yes, or chose 'X' and said yes
    
    while True:
        # Print the board
        print_board(board)
        
        # Check if it's the player's turn
        if player_turn:
            player_move(board, player_choice)
            if check_win(board, player_choice):
                print_board(board)
                print(f"Congratulations, you win!")
                break
            if is_board_full(board):
                print_board(board)
                print("It's a draw!")
                break
            player_turn = False  # Switch to AI's turn
        else:
            # AI's turn using random move
            print(f"AI ({ai_choice})'s move...")
            ai_move_position = ai_move(board)
            board[ai_move_position] = ai_choice
            if check_win(board, ai_choice):
                print_board(board)
                print(f"AI ({ai_choice}) wins! Better luck next time.")
                break
            if is_board_full(board):
                print_board(board)
                print("It's a draw!")
                break
            player_turn = True  # Switch to player's turn

# Start the game
if __name__ == "__main__":
    main()

----------------------------------------------------------------------------------------------

# 8Puzzle

import heapq

# Manhattan distance heuristic function
def manhattan_distance(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:  # Ignore the empty tile (0)
                x, y = divmod(state[i][j] - 1, 3)  # Get goal position
                distance += abs(x - i) + abs(y - j)
    return distance

# Misplaced tiles heuristic function
def misplaced_tiles(state, goal):
    return sum(state[i][j] != goal[i][j] and state[i][j] != 0 for i in range(3) for j in range(3))

# Function to get the neighbors of the current state
def get_neighbors(state):
    neighbors = []
    x, y = [(i, row.index(0)) for i, row in enumerate(state) if 0 in row][0]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right moves
    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < 3 and 0 <= new_y < 3:  # Ensure valid move
            new_state = [list(row) for row in state]
            new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
            neighbors.append(new_state)

    return neighbors

# Function to reconstruct the path from the start to the goal state
def reconstruct_path(came_from, current, misplaced_values):
    path = []
    while current in came_from:
        path.append((current, misplaced_values[current]))
        current = came_from[current]
    path.reverse()
    return path

# A* Algorithm to solve 8-puzzle problem
def a_star(start, goal):
    # Priority queue for A* search
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {str(start): 0}
    f_score = {str(start): manhattan_distance(start)}
    misplaced_values = {str(start): misplaced_tiles(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, str(current), misplaced_values)

        for neighbor in get_neighbors(current):
            tentative_g_score = g_score[str(current)] + 1

            if str(neighbor) not in g_score or tentative_g_score < g_score[str(neighbor)]:
                came_from[str(neighbor)] = str(current)
                g_score[str(neighbor)] = tentative_g_score
                f_score[str(neighbor)] = tentative_g_score + manhattan_distance(neighbor)
                misplaced_values[str(neighbor)] = misplaced_tiles(neighbor, goal)
                heapq.heappush(open_set, (f_score[str(neighbor)], neighbor))

    return None  # No solution found

# Function to print the puzzle state along with its heuristic value and misplaced tiles
def print_puzzle(state, heuristic=None, goal=None):
    print("Puzzle State:")
    for row in state:
        print(' '.join(map(str, row)))
    if heuristic is not None:
        print(f"Heuristic (Misplaced Tiles): {heuristic} tiles misplaced")

    if goal is not None:
        misplaced = []
        for i in range(3):
            for j in range(3):
                if state[i][j] != goal[i][j] and state[i][j] != 0:
                    misplaced.append(f"Tile {state[i][j]} at ({i}, {j})")
        if misplaced:
            print("Misplaced Tiles: " + ", ".join(misplaced))
        else:
            print("All tiles are correctly placed!")
    print()

# Function to get input from the user
def get_input():
    print("Enter the start state (enter 0 for the empty space):")
    start = []
    for i in range(3):
        row = list(map(int, input(f"Row {i+1}: ").split()))
        start.append(row)

    print("\nEnter the goal state (enter 0 for the empty space):")
    goal = []
    for i in range(3):
        row = list(map(int, input(f"Row {i+1}: ").split()))
        goal.append(row)

    return start, goal

# Main function
def main():
    start, goal = get_input()

    print("\nInitial State:")
    print_puzzle(start, goal=goal)

    solution = a_star(start, goal)

    if solution:
        print("Solution found! Steps:")
        step_number = 1
        for step, heuristic in solution:
            print(f"Step {step_number}:")
            print_puzzle(eval(step), heuristic, goal)
            step_number += 1
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()


----------------------------------------------------------------------------------------------

# DecisionTree

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier using ID3 (using entropy as criterion for information gain)
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the ID3 Decision Tree on the test set: {accuracy * 100:.2f}%")

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.show()

# Convert the new sample to a DataFrame with proper feature names
new_sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)

# Predict the class
predicted_class = clf.predict(new_sample)
print(f"The predicted class for the sample {new_sample.values.tolist()} is: {iris.target_names[predicted_class[0]]}")


----------------------------------------------------------------------------------------------

# LinearRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the sample data
data = pd.DataFrame({
    'Area': [1500, 1700, 2400, 3000, 3500],
    'Price': [300000, 350000, 500000, 600000, 700000]
})

# Independent and dependent variables
X = data['Area']
Y = data['Price']

# Number of data points
N = len(X)

# Calculate slope (m) and intercept (c)
m = (N * sum(X * Y) - sum(X) * sum(Y)) / (N * sum(X ** 2) - (sum(X) ** 2))
c = (sum(Y) - m * sum(X)) / N

print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")

# Predicting Y values based on the regression line
Y_pred = m * X + c

# Plotting the data points
plt.scatter(X, Y, color='blue', label='Data points')

# Plotting the regression line
plt.plot(X, Y_pred, color='red', label=f'Regression line (Y = {m:.2f}X + {c:.2f})')

# Adding labels and title
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

----------------------------------------------------------------------------------------------

# NaiveBayes

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

def calculate_prior(y_train):
    classes, counts = np.unique(y_train, return_counts=True)
    return {cls: counts[i] / len(y_train) for i, cls in enumerate(classes)}

def calculate_likelihoods(X_train, y_train):
    likelihoods = {}
    for cls in np.unique(y_train):
        X_class = X_train[y_train == cls]
        likelihoods[cls] = {
            feature_index: {value: (count / len(X_class)) for value, count in zip(*np.unique(X_class[:, feature_index], return_counts=True))}
            for feature_index in range(X_train.shape[1])
        }
    return likelihoods

def predict(X_test, priors, likelihoods):
    y_pred = []
    for x in X_test:
        class_probs = {cls: priors[cls] * np.prod([likelihoods[cls][i].get(x[i], 1e-6) for i in range(len(x))]) for cls in priors}
        y_pred.append(max(class_probs, key=class_probs.get))
    return y_pred

# Load and preprocess data
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target
data = data.apply(lambda col: pd.cut(col, bins=3, labels=False) if col.name != 'target' else col)

# Train-test split
X = data.drop(columns=['target']).values
y = data['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and predict
priors = calculate_prior(y_train)
likelihoods = calculate_likelihoods(X_train, y_train)
y_pred = predict(X_test, priors, likelihoods)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


------------------the end of code 1--------------------------------------------------------------------

Easier Code for NaiveBayes:

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to load dataset from a CSV file
def load_custom_dataset(filepath):
    data = pd.read_csv(filepath)
    X = data.iloc[:, :-1].values  # Features: all columns except the last one
    y = data.iloc[:, -1].values   # Target: last column
    return X, y

# Prompt user for the dataset file path
filepath = input("Enter the path to the dataset CSV file: ")

try:
    # Load the dataset
    X, y = load_custom_dataset(filepath)

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Gaussian Naive Bayes classifier
    nb_classifier = GaussianNB()

    # Train the classifier on the training data
    nb_classifier.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = nb_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of Naive Bayes Classifier: {accuracy * 100:.2f}%")

    # Display a detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Display the confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

except Exception as e:
    print(f"An error occurred while loading or processing the dataset: {e}")


----------end of code 2------------------------------------------------------------------------------------

#Using iris
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: species (0: setosa, 1: versicolor, 2: virginica)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the classifier on the training data
nb_classifier.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Naive Bayes Classifier: {accuracy * 100:.2f}%")

# Display a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

                   
----------------------------------------------------------------------------------------------

# KNN

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: species (0: setosa, 1: versicolor, 2: virginica)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the k-NN classifier with k=3
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of k-NN with k={k}: {accuracy * 100:.2f}%")

# Display a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
