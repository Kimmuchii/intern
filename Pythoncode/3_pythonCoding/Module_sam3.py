from ultralytics import SAM

# 1. Use 'sam2_b.pt' or 'sam_b.pt'. Ultralytics will auto-download it for you!
model = SAM("sam2_b.pt")

# 2. Run inference on a sample image
results = model.predict(
    source="https://ultralytics.com", 
    bboxes=[100, 100, 300, 300]  # Standard geometric tracking box
)

# 3. View the results on your screen
results.show()
