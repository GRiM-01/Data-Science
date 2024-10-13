import csv
import os
import sys
import random  
import numpy as np 
import matplotlib.pyplot as plt  
from typing import TypeVar, NamedTuple, List, Dict 
from collections import Counter, defaultdict   
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

X = TypeVar('X')  # generic type to represent a data point  
Vector = List[float]
class LabeledPoint(NamedTuple):  
     point: Vector
     label: str  


def get_script_directory():
    return os.path.dirname(sys.argv[0])

script_directory = get_script_directory()

# open csv file
file = os.path.join(script_directory, "iris_dirty.csv")
with open(file, 'r') as f:  
    data = list(csv.reader(f, delimiter=",")) 
data = np.array(data)  
# print(len(data))

original_species_counts = Counter(row[5] for row in data)
# print("Original species counts:", original_species_counts)

def try_parse_row(row:List[str])-> tuple[float, float, float, float, str]:     
    _,sepal_length, sepal_width, petal_length, petal_width, species = row     
    try:  
            float (sepal_length)     
    except ValueError:         
        return None  
            
    try:  
            float (sepal_width)     
    except ValueError:         
        return None  
            
    try:  
            float (petal_length)     
    except ValueError:         
        return None  
            
    try:  
            float (petal_width)     
    except ValueError:         
        return None 
    
    try:  
            str(species)     
    except ValueError:     
        return None  
       
    else:  
        return (float (sepal_length),float(sepal_width), float(petal_length),float(petal_width),str(species).lower()) 

def distance(v, w):
    return np.linalg.norm(np.array(v) - np.array(w))


def knn_classify(k: int, labeled_points: List[LabeledPoint], new_point: Vector) -> str:  
    # Order the labeled points from nearest to farthest.  
    by_distance = sorted(labeled_points,   
    key=lambda lp: distance(lp.point, new_point))  
    k_nearest_labels = [lp.label for lp in by_distance[:k]]  
    return majority_vote(k_nearest_labels)  
 
def parse_iris_row(row: List[str]) -> LabeledPoint:  
    measurements = [float(value) for value in row[:-1]]     
    label = row[-1].split("-")[-1]                  
    return LabeledPoint(measurements, label)

def majority_vote (labels: List[str])->str:      
    vote_counts = Counter(labels)  
    winner, winner_count = vote_counts.most_common(1)[0]     
    num_winners = len([count for count in vote_counts.values() if count == winner_count])     
    if num_winners == 1:  
        # print(winner)
        return winner
    else:  
        return majority_vote(labels[:-1])
    
def split_data(data: List[X], prob: float) -> tuple[List[X], List[X]]:  
    """Split data into fractions [prob, 1 - prob]"""     
    data_copy = data[:]                    # Make a shallow copy because shuffle modifies the list.    
    random.shuffle(data_copy)               
    cut = int(len(data_copy) * prob)        
    return data_copy[:cut], data_copy[cut:]     # and split the shuffled list there.  


clean_data = []
dirty_data = []  
for row in data:  
    maybe_iris = try_parse_row(row)     
    if maybe_iris is None:  
        print(f"skipping invalid row: {row}")    
        dirty_data.append(row)
    else:  
        clean_data.append(maybe_iris)  

print("\nDirty data:",len(dirty_data))
print("Clean data:",len(clean_data))

train, test = split_data(clean_data, 0.75)  
print("Number of training points:", len(train), "Number of testing point:", len(test)) 

majority_vote(['a', 'b', 'c', 'b','a']) 

iris_data = [parse_iris_row(row) for row in clean_data]  
          
points_by_species: Dict[str, List[Vector]] = defaultdict(list) 
for iris in iris_data: points_by_species[iris.label].append(iris.point)  
 
metrics = ['sepal length', 'sepal width', 'petal length', 'petal width']     
pairs = [(i, j) for i in range(4) for j in range(4) if i < j]     
marks = ['+', '.', 'x']
      
fig, ax = plt.subplots(2, 3) 

for row in range(2):         
    for col in range(3):  
        i, j = pairs[3 * row + col]  
        ax[row][col].set_title(f"{metrics[i]} vs {metrics[j]}", fontsize=8)             
        ax[row][col].set_xticks([])             
        ax[row][col].set_yticks([])

        for mark, (species, points) in zip(marks, points_by_species.items()):  
            xs = [point[i] for point in points]                 
            ys = [point[j] for point in points]  
            ax[row][col].scatter(xs, ys, marker=mark, label=species)  
            ax[row][col].legend(['setosa', 'versicolor', 'virginica'], loc='lower right', prop={'size': 6})
  
# print(clean_data)

# split the data to training and testing data 
random.seed(12) 
iris_train, iris_test = split_data(iris_data, 0.70)  

# Track how many times we see predicted vs actual
confusion_matrix_counts = defaultdict(int)     
num_correct = 0  

for iris in iris_test:
    predicted = knn_classify(5, iris_train, iris.point)         
    actual = iris.label  
    confusion_matrix_counts[(predicted, actual)] += 1  
    if predicted == actual:             
        num_correct += 1  

# Calculate the overall accuracy
pct_correct = num_correct / len(iris_test)     
print("Accuracy:", pct_correct)

# Create confusion matrix array
labels = sorted(set([label for _, label in confusion_matrix_counts.keys()]))
num_classes = len(labels)

# Initialize confusion matrix with zeros
cm = np.zeros((num_classes, num_classes), dtype=int)

label_to_index = {label: idx for idx, label in enumerate(labels)}

for (pred, actual), count in confusion_matrix_counts.items():
    cm[label_to_index[pred]][label_to_index[actual]] = count

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()