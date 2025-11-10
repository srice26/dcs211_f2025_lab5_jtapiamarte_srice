import random 
import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)
import seaborn as sns   # yay for Seaborn plots!
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier


###########################################################################

def drawDigitHeatmap(pixels: np.ndarray, showNumbers: bool = True) -> None:
    ''' Draws a heat map of a given digit based on its 8x8 set of pixel values.
    Parameters:
        pixels: a 2D numpy.ndarray (8x8) of integers of the pixel values for
                the digit
        showNumbers: if True, shows the pixel value inside each square
    Returns:
        None -- just plots into a window
    '''

    (fig, axes) = plt.subplots(figsize = (4.5, 3))  # aspect ratio

    rgb = (0, 0, 0.5)  # each in (0,1), so darkest will be dark blue
    colormap = sns.light_palette(rgb, as_cmap=True)    
    #all seaborn palettes: 
    #https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f

    #plot the heatmap; https://seaborn.pydata.org/generated/seaborn.heatmap.html
    #(fmt = "d" indicates to show annotation with integer format)
    sns.heatmap(pixels, annot = showNumbers, fmt = "d", linewidths = 0.5, \
                ax = axes, cmap = colormap)
    plt.show(block = False)

def fetchDigit(df: pd.core.frame.DataFrame, which_row: int) -> tuple[int, np.ndarray]:
    ''' For digits.csv data represented as a dataframe, this fetches the digit from
        the corresponding row, reshapes, and returns a tuple of the digit and a
        numpy array of its pixel values.
    Parameters:
        df: pandas data frame expected to be obtained via pd.read_csv() on digits.csv
        which_row: an integer in 0 to len(df)
    Returns:
        a tuple containing the represented digit and a numpy array of the pixel
        values
    '''
    digit  = int(round(df.iloc[which_row, 64]))
    pixels = df.iloc[which_row, 0:64]   # don't want the rightmost rows
    pixels = pixels.values              # converts to numpy array
    pixels = pixels.astype(int)         # convert to integers for plotting
    pixels = np.reshape(pixels, (8,8))  # makes 8x8
    return (digit, pixels)              # return a tuple


def cleanTheData(df: pd.DataFrame) -> np.ndarray:
    '''
    Cleans the digits.csv dataframe for k-NN.
    Parameters:
        df: The digits.csv data loaded with pd.read_csv().
    Returns:
        np.ndarray: Array of the cleaned data.
    '''
    import numpy as np
    import pandas as pd

    #define label and pixel columns 
    label_col = "actual_digit"
    if label_col not in df.columns:
        raise ValueError(f"Expected label column '{label_col}' not found in dataframe.")
    pixel_cols = [c for c in df.columns if c != label_col]

    #ensure numeric conversion without nuking good values
    df[pixel_cols] = df[pixel_cols].apply(pd.to_numeric, errors="coerce")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    #drop rows where the label is missing
    df = df.dropna(subset=[label_col]).copy()
    df[pixel_cols] = df[pixel_cols].fillna(0)
    df[pixel_cols] = df[pixel_cols] / 16.0
    #reorder columns so label comes first
    df = df[[label_col] + pixel_cols]
    all_arr = df.to_numpy()
    
    return all_arr

def predictiveModel(train_arr: np.ndarray, features: np.ndarray) -> int:
    '''
    Implements a 1-NN classifier for one test sample.
    Parameters:
        train_arr: Training data where the first column is the label
                   and the rest are pixel values.
        features: The pixel values for one test digit
    Returns:
        int: predicted digit label (0-9)
    '''
    #separate labels and pixels
    labels = train_arr[:, 0].astype(int)
    pixels = train_arr[:, 1:]  # (N, 64)
    #ensure the test sample has same dimensionality
    if features.ndim != 1:
        features = features.flatten()
    if features.shape[0] != pixels.shape[1]:
        raise ValueError(f"Feature vector length {features.shape[0]} does not match training data ({pixels.shape[1]} pixels).")

    #compute distances
    dists = np.linalg.norm(pixels - features, axis=1)
    #return label of nearest neighbor
    nn_index = np.argmin(dists)
    return int(labels[nn_index])

def splitData(all_data):
    '''
    Input: numpy array (all_data)
           column 0 = labels
           columns 1.. = features
    Output: X_test, y_test, X_train, y_train
    '''
    y = all_data[:, 0]          #labels
    X = all_data[:, 1:]         #pixel data

    split_index = int(0.8 * len(all_data))

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_test, y_test, X_train, y_train

def compareLabels(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> int:
    '''A neatly formatted comparison, returning the number correct.'''
    num_labels = len(predicted_labels)
    num_correct = 0

    for i in range(num_labels):
        predicted = int(round(predicted_labels[i]))  # protects from float imprecision
        actual    = int(round(actual_labels[i]))
        result = "incorrect"
        if predicted == actual:
            result = ""
            num_correct += 1

        # Formatting stays identical, but now we print the digit itself
        print(f"row {i:>3d} : ", end="")
        print(f"{str(predicted):>12s} ", end="")   # predicted digit as string
        print(f"{str(actual):<12s}   {result}")    # actual digit as string

    print()
    print(f"Correct: {num_correct} out of {num_labels}")
    return num_correct

def runSimpleKNN(X_train, y_train, X_test, y_test, k=3):
    '''
    Runs scikit-learn k-NN once using a guessed value of k.
    Prints accuracy and uses compareLabels to display results.
    '''

    model = KNeighborsClassifier(n_neighbors=k) #training the model
    model.fit(X_train, y_train)

    predictions = model.predict(X_test) #predictions for test data

    accuracy = (predictions == y_test).mean() #compute accuracy of predictions
    print("Accuracy:", round(accuracy, 3))

    compareLabels(predictions, y_test) #compare predicted vs actual labels

    return predictions, accuracy

def findBestK(X_train: np.ndarray, y_train: np.ndarray) -> int:
    '''
    Determines the best k for k-NN by testing multiple k values on the training set.
    Returns the k with the highest training accuracy.
    '''

    k_values = [1, 3, 5, 7, 9]
    best_k = None
    best_accuracy = -1

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        preds = model.predict(X_train)
        accuracy = (preds == y_train).mean()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    return best_k

def trainAndTest(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, best_k: int):
    '''
    Trains k-NN using the best k and returns predictions on X_test.
    '''
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    return predictions


# Example usage (to be placed inside main):
# random.seed(8675309)
# k1 = findBestK(X_train, y_train)
# random.seed(5551212)
# k2 = findBestK(X_train, y_train)
# random.seed(12345)
# k3 = findBestK(X_train, y_train)
# best_k = k1
# final_predictions = trainAndTest(X_train, y_train, X_test, best_k)
# final_accuracy = (final_predictions == y_test).mean()


################################################################

def main() -> None:
    # for read_csv, use header=0 when row 0 is a header row
    filename = 'digits.csv'
    df = pd.read_csv(filename, header = 0)
    print(df.head())
    print("Shape:", df.shape, list(df.columns))
    print(f"{filename} : file read into a pandas dataframe...")

    num_to_draw = 5
    for i in range(num_to_draw):
        # let's grab one row of the df at random, extract/shape the digit to be
        # 8x8, and then draw a heatmap of that digit
        random_row = random.randint(0, len(df) - 1)
        (digit, pixels) = fetchDigit(df, random_row)

        print(f"The digit is {digit}")
        print(f"The pixels are\n{pixels}")  
        drawDigitHeatmap(pixels)
        plt.show()
    all_arr = cleanTheData(df)

    # 1-NN classifier
    from progress.bar import Bar
    N = all_arr.shape[0]
    cut = int(0.8 * N)
    train, test = all_arr[:cut], all_arr[cut:]
    y_test = test[:, 0].astype(int)
    X_test = test[:, 1:]

    print("Running 1-NN on 80/20 split...")
    correct = 0
    bar = Bar('Predicting', max=len(X_test))
    for i in range(len(X_test)):
        y_hat = predictiveModel(train, X_test[i])
        if y_hat == y_test[i]:
            correct += 1
        bar.next()
    bar.finish()
    acc = correct / len(y_test)
    print(f"\nAccuracy (80/20 split): {acc:.3f}\n")

    # swap split (20/80)
    cut = int(0.2 * N)
    test, train = all_arr[:cut], all_arr[cut:]
    y_test = test[:, 0].astype(int)
    X_test = test[:, 1:]

    print("Running 1-NN on 20/80 swap")
    correct = 0
    bar = Bar('Predicting (swap)', max=len(X_test))
    for i in range(len(X_test)):
        y_hat = predictiveModel(train, X_test[i])
        if y_hat == y_test[i]:
            correct += 1
        bar.next()
    bar.finish()
    acc2 = correct / len(y_test)
    print(f"\nAccuracy (20/80 swap): {acc2:.3f}\n")

#--------------------------------------------------------------------
    all_data = cleanTheData(df)

    # 2. Split data (Part 7)
    print("Splitting data into train/test...")
    X_test, y_test, X_train, y_train = splitData(all_data)

    # 3. Run simple k-NN test with guessed k (Part 8)
    print("Running simple k-NN with guessed k ===")
    guessed_k = 3
    preds8, acc8 = runSimpleKNN(X_train, y_train, X_test, y_test, guessed_k)

    # 4. Find best k for three seeds (Part 9)
    
    print("Finding best k using multiple seeds ===")
    random.seed(8675309)
    k1 = findBestK(X_train, y_train)
    print(f"Best k for seed 8675309: {k1}")

    random.seed(5551212)
    k2 = findBestK(X_train, y_train)
    print(f"Best k for seed 5551212: {k2}")

    my_seed = 12345
    random.seed(my_seed)
    k3 = findBestK(X_train, y_train)
    print(f"Best k for your seed {my_seed}: {k3}")

    # Choose the best k (can justify using first, majority, or highest performance)
    best_k = k1
    print(f"best_k: {best_k}")

    # 5. Train with best k and test (Part 10)
    print("Training and testing with best_k ===")
    final_predictions = trainAndTest(X_train, y_train, X_test, best_k)
    final_accuracy = (final_predictions == y_test).mean()
    print(f"Final accuracy using best_k={best_k}: {round(final_accuracy, 3)}")

    # Compare predictions
    compareLabels(final_predictions, y_test)

    print("Done.")


    

if __name__ == "__main__":
    main()
