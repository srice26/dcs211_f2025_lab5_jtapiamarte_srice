import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)
import seaborn as sns   # yay for Seaborn plots!
import matplotlib.pyplot as plt
import random

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
    # all seaborn palettes: 
    # https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f

    # plot the heatmap;  see: https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # (fmt = "d" indicates to show annotation with integer format)
    sns.heatmap(pixels, annot = showNumbers, fmt = ".1f", linewidths = 0.5, \
                ax = axes, cmap = colormap)
    plt.show(block = False)

###########################################################################
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
    """Cleans the digits.csv dataframe for k-NN.
    
    Parameters:
        df: The digits.csv data loaded with pd.read_csv().
    
    Returns:
        np.ndarray: Array of the cleaned data.
    """
    # Drop extra column
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    if 'excerpted from http://yann.lecun.com/exdb/mnist/' in df.columns:
        df = df.drop(columns=['excerpted from http://yann.lecun.com/exdb/mnist/'])

    # Define label and pixel columns explicitly
    label_col = "actual_digit"
    if label_col not in df.columns:
        raise ValueError(f"Expected label column '{label_col}' not found in dataframe.")
    pixel_cols = [c for c in df.columns if c != label_col]
    
    # Ensure numeric conversion without nuking good values
    df[pixel_cols] = df[pixel_cols].apply(pd.to_numeric, errors="coerce")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    # Drop only rows where the label is missing
    df = df.dropna(subset=[label_col]).copy()
    df[pixel_cols] = df[pixel_cols].fillna(0)
    df[pixel_cols] = df[pixel_cols] / 16.0
    # Reorder columns so label comes first
    df = df[[label_col] + pixel_cols]
    all_arr = df.to_numpy()
    
    return all_arr

def predictiveModel(train_arr: np.ndarray, features: np.ndarray) -> int:
    """Implements a 1-NN classifier for one test sample.
    Parameters:
        train_arr: Training data where the first column is the label
                   and the rest are pixel values.
        features: The pixel values for one test digit
    Returns:
        int: predicted digit label (0–9)
    """
    # Separate labels and pixels
    labels = train_arr[:, 0].astype(int)
    pixels = train_arr[:, 1:]  # (N, 64)
    # Ensure the test sample has same dimensionality
    if features.ndim != 1:
        features = features.flatten()
    if features.shape[0] != pixels.shape[1]:
        raise ValueError(f"Feature vector length {features.shape[0]} does not match training data ({pixels.shape[1]} pixels).")

    # Compute distances
    dists = np.linalg.norm(pixels - features, axis=1)
    # Return label of nearest neighbor
    nn_index = np.argmin(dists)
    return int(labels[nn_index])

###################
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
    print("\n✅ Data cleaned and converted to NumPy array.")
    print(f"Shape: {all_arr.shape} (rows × cols)\n")

    # 1-NN classifier
    from progress.bar import Bar
    print(f"all_arr: {all_arr}")
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
    print(y_test,cut,train,test)
    acc = correct / len(y_test)
    print(f"\nAccuracy (80/20 split): {acc:.3f}\n")

    # swap split (20/80)
    cut = int(0.2 * N)
    test, train = all_arr[:cut], all_arr[cut:]
    y_test = test[:, 0].astype(int)
    print(f"y_test: {y_test}")
    print(f"debug check: { np.unique(all_arr[:,0])}")
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
    # Visualize first five incorrectly predicted digits
    shown = 0
    y_test = test[:, 0].astype(int)
    X_test = test[:, 1:]
    for i in range(len(X_test)):
        y_pred = predictiveModel(train, X_test[i])
        if y_pred != y_test[i]:
            pixels = np.reshape(X_test[i] * 16, (8,8))
            drawDigitHeatmap(pixels)
            plt.show(block=False)
            shown += 1
            if shown == 5:
                break
    #
    # OK!  Onward to knn for digits! (based on your iris work...)
    #

###############################################################################
# wrap the call to main inside this if so that _this_ file can be imported
# and used as a library, if necessary, without executing its main
if __name__ == "__main__":
    main()
