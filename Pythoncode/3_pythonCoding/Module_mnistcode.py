import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

print("🔄 Fetching MNIST dataset from OpenML (this may take a few seconds)...")
# Load the dataset (784 features represents a flattened 28x28 pixel grid)
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist["data"], mnist["target"]

print(f"✅ Dataset loaded! Image array shape: {X.shape}")
print(f"✅ Total Labels found: {y.shape}\n")

# Set up a 1 row by 5 column grid to preview images
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
fig.suptitle("MNIST Dataset Samples", fontsize=14, fontweight='bold')

for i in range(5):
    # 1. Grab a target pixel array and reshape it back to a 28x28 square image
    digit_image = X[i].reshape(28, 28)
    
    # 2. Render the pixel array onto the specific subplot
    axes[i].imshow(digit_image, cmap="gray")
    axes[i].set_title(f"Label: {y[i]}")
    axes[i].axis("off") # Remove axis numbers for a cleaner look

print("🖼️ Displaying image grid window...")
plt.tight_layout()
plt.show()



