# ==============================================================================
# TS-CrowdNet: A Temporal-Spatial Framework for Crowd Counting
#
# This script transforms a spatial-only crowd counting model into a
# temporal-spatial model (TS-CrowdNet). It processes video frames as
# sequences to leverage temporal information for more accurate predictions.
#
# Key Changes from Original Code:
# 1. Data Preparation: Switched from ImageDataGenerator to a custom sequence
#    generator to handle ordered frames.
# 2. Model Architecture: Wrapped the ResNet50 backbone in a TimeDistributed
#    layer and added an LSTM layer to process temporal data.
# 3. Training: Adapted the training loop to use the new sequence generator.
# ==============================================================================

# --- 1. Introduction: Imports and Setup ---
import os
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools

# TensorFlow and Keras Imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set plotting style
sns.set(style='whitegrid', context='notebook', palette='deep')
np.random.seed(42) # Set seed for reproducibility


# --- Helper Function for Plotting ---
def add_one_to_one_correlation_line(ax, **plot_kwargs):
    """Adds a 1:1 correlation line to a scatter plot."""
    lim_min, lim_max = pd.DataFrame([ax.get_ylim(), ax.get_xlim()]).agg({0: 'min', 1: 'max'})
    plot_kwargs_internal = dict(color='grey', ls='--')
    plot_kwargs_internal.update(plot_kwargs)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], **plot_kwargs_internal)
    ax.set_ylim([lim_min, lim_max])
    ax.set_xlim([lim_min, lim_max])


# --- 2. Data Preparation for Temporal Sequences ---

# --- 2.1 Load and Sort Data ---
# Load the labels and ensure they are sorted chronologically by 'id'
# This is CRITICAL for creating meaningful time sequences.
df = pd.read_csv("/root/.cache/kagglehub/datasets/fmena14/crowd-counting/versions/3/labels.csv")
df['image_name'] = df['id'].map('seq_{:06d}.jpg'.format)
df = df.sort_values('id').reset_index(drop=True)

print("Data loaded and sorted:")
print(df.head())

# --- 2.2 Create Sequences of File Paths ---
data_dir = '/root/.cache/kagglehub/datasets/fmena14/crowd-counting/versions/3/frames/frames'
all_image_paths = [os.path.join(data_dir, fname) for fname in df['image_name']]
all_counts = df['count'].values

SEQUENCE_LENGTH = 5  # Use 4 frames to predict the count in the 5th frame
X_sequence_paths, y_sequence_counts = [], []

# This loop creates overlapping sequences of file paths
for i in range(len(all_image_paths) - SEQUENCE_LENGTH):
    X_sequence_paths.append(all_image_paths[i : i + SEQUENCE_LENGTH])
    # The target is the count of the LAST frame in the sequence
    y_sequence_counts.append(all_counts[i + SEQUENCE_LENGTH - 1])

print(f"\nCreated {len(X_sequence_paths)} sequences of length {SEQUENCE_LENGTH}.")


# --- 2.3 Custom Data Generator for Sequences ---
# We create a custom generator because ImageDataGenerator shuffles individual
# images, which breaks the temporal order.
def sequence_generator(image_paths_list, counts_list, batch_size, img_size):
    """
    Yields batches of image sequences and their corresponding counts.
    """
    num_samples = len(image_paths_list)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for offset in range(0, num_samples, batch_size):
            batch_indices = indices[offset:offset + batch_size]
            
            batch_X, batch_y = [], []
            for i in batch_indices:
                sequence_paths = image_paths_list[i]
                target_count = counts_list[i]
                
                sequence_images = []
                for img_path in sequence_paths:
                    img = load_img(img_path, target_size=(img_size, img_size))
                    img_array = img_to_array(img)
                    img_array = resnet50.preprocess_input(img_array)
                    sequence_images.append(img_array)
                
                batch_X.append(sequence_images)
                batch_y.append(target_count)
                
            yield np.array(batch_X), np.array(batch_y)

# --- 2.4 Split Data and Create Generators ---
train_split_idx = int(0.8 * len(X_sequence_paths))
X_train_paths, X_val_paths = X_sequence_paths[:train_split_idx], X_sequence_paths[train_split_idx:]
y_train, y_val = y_sequence_counts[:train_split_idx], y_sequence_counts[train_split_idx:]

IMG_SIZE = 224
BATCH_SIZE = 16 # Use a smaller batch size as sequences consume more memory

train_gen = sequence_generator(X_train_paths, y_train, BATCH_SIZE, IMG_SIZE)
valid_gen = sequence_generator(X_val_paths, y_val, BATCH_SIZE, IMG_SIZE)

print(f"\nTraining sequences: {len(X_train_paths)}")
print(f"Validation sequences: {len(X_val_paths)}")


# --- 3. TS-CrowdNet Model ---

# --- 3.1 Build the Temporal-Spatial Model ---
def build_ts_crowdnet_model(sequence_length, img_size):
    """
    Builds the TS-CrowdNet model architecture.
    """
    # Spatial Backbone: Pre-trained ResNet50
    base_model = resnet50.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3),
        pooling='avg'
    )
    # Make the base model non-trainable to use its learned features
    base_model.trainable = False

    # Define the input shape for a sequence of images
    sequence_input = Input(shape=(sequence_length, img_size, img_size, 3))

    # Wrap the base model in a TimeDistributed layer. This applies the ResNet50
    # model to each frame of the input sequence independently.
    time_distributed_features = TimeDistributed(base_model)(sequence_input)

    # Temporal Module: An LSTM layer processes the sequence of feature vectors
    # to learn temporal patterns.
    lstm_output = LSTM(512, dropout=0.5)(time_distributed_features)

    # Regression Head: A dense layer to interpret the LSTM output, followed by
    # the final prediction layer.
    x = Dense(256, activation='relu')(lstm_output)
    predictions = Dense(1, activation='linear')(x)

    model = Model(inputs=sequence_input, outputs=predictions, name="TS_CrowdNet")
    return model

# Instantiate the model
ts_model = build_ts_crowdnet_model(SEQUENCE_LENGTH, IMG_SIZE)


# --- 3.2 Compile the Model ---
optimizer = Adam(learning_rate=0.001)
ts_model.compile(
    optimizer=optimizer,
    loss="mean_squared_error",
    metrics=['mean_absolute_error']
)

ts_model.summary()


# --- 4. Train the Model ---
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_loss',
    patience=3,
    verbose=1,
    factor=0.2,
    min_lr=0.000001
)

# Calculate steps for the generator
steps_per_epoch = len(X_train_paths) // BATCH_SIZE
validation_steps = len(X_val_paths) // BATCH_SIZE

print("\n--- Starting Model Training ---")
history = ts_model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=valid_gen,
    validation_steps=validation_steps,
    verbose=2,
    callbacks=[learning_rate_reduction]
)
print('\n--- Training Done ---')


# --- 5. Evaluate the Model ---

# --- 5.1 Plot Training and Validation Curves ---
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(history.history['loss'], color='b', label="Training loss")
ax.plot(history.history['val_loss'], color='r', label="Validation loss")
ax.set_ylim(top=np.max(history.history['val_loss']) * 1.2, bottom=0)
ax.set_title("Training and Validation Loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss (MSE)")
ax.legend(loc='best', shadow=True)
plt.show()

# --- 5.2 Predict on Validation Set and Analyze ---
print("\n--- Predicting on Validation Set ---")
# Create a fresh generator for validation to ensure order
val_gen_eval = sequence_generator(X_val_paths, y_val, BATCH_SIZE, IMG_SIZE)

all_labels = []
all_pred = []

for i in range(validation_steps):
    x, y = next(val_gen_eval)
    pred_i = ts_model.predict(x)[:, 0]
    all_labels.extend(y)
    all_pred.extend(pred_i)

df_predictions = pd.DataFrame({'True values': all_labels, 'Predicted values': all_pred})

# --- 5.3 Visualize Predictions vs. True Values ---
ax = df_predictions.plot.scatter('True values', 'Predicted values', alpha=0.5, s=14, figsize=(9, 9))
ax.grid(axis='both')
add_one_to_one_correlation_line(ax)
ax.set_title('Validation Set: Predicted vs. True Values')
plt.show()

# --- 5.4 Calculate Final Metrics ---
# This section was already present but is kept for clarity
mse_final = mean_squared_error(df_predictions['True values'], df_predictions['Predicted values'])
pearson_r = sc.stats.pearsonr(df_predictions['True values'], df_predictions['Predicted values'])[0]

print(f'\nIntermediate Validation Metrics:')
print(f'Mean Squared Error (MSE): {mse_final:.2f}')
print(f'Pearson Correlation (r): {pearson_r:.3f}')


# --- 5.5 Final Performance Metrics Summary ---
# This new section explicitly calculates and prints Accuracy, MAE, and MSE.

print("\n" + "="*50)
print("--- Final Performance Metrics Summary ---")
print("="*50)

# Retrieve true values and predictions
true_values = df_predictions['True values'].values
predicted_values = df_predictions['Predicted values'].values

# 1. Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(true_values, predicted_values)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print("-> This means, on average, the model's prediction is off by ~{:.0f} people.".format(mae))

# 2. Calculate Mean Squared Error (MSE)
mse = mean_squared_error(true_values, predicted_values)
print(f"\nMean Squared Error (MSE): {mse:.4f}")
print("-> This metric penalizes larger errors more heavily.")

# 3. Calculate Accuracy (for regression context)
# We define "accurate" as a prediction that is within a certain percentage
# (e.g., 20%) of the true value. This is a common way to frame accuracy
# for a regression task.
tolerance = 0.20  # 20% tolerance
accurate_predictions = np.sum(np.abs(true_values - predicted_values) / true_values <= tolerance)
total_predictions = len(true_values)
accuracy = (accurate_predictions / total_predictions) * 100

print(f"\nAccuracy (within {tolerance:.0%} tolerance): {accuracy:.2f}%")
print(f"-> This means {accuracy:.2f}% of the predictions were within {tolerance:.0%} of the actual crowd count.")
print("="*50)

