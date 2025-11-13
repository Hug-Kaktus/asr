import matplotlib.pyplot as plt

steps = list(range(10, 110, 10))
steps_end = list(range(40900, 41000, 10))
loss_values = [340.35, 270.23, 256.92, 234.10, 228.38, 206.66, 197.29, 192.14, 180.71, 171.73]

loss_values_end = [15.34, 15.31, 15.24, 15.18, 15.17, 15.09, 15.04, 15.03, 15.04, 15.03]

plt.figure(figsize=(8, 5))
plt.plot(steps_end, loss_values_end, marker='o', linestyle='-', color='b', linewidth=2)

plt.title('Neural Network Training Loss last 100 steps', fontsize=14)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Loss', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)

for i, loss in enumerate(loss_values_end):
    plt.text(steps[i], loss + 0.02, f"{loss:.2f}", ha='center', fontsize=12)

plt.tight_layout()
plt.show()
