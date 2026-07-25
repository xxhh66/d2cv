import os
# 普通 SGD 与 Momentum SGD 更新过程演示

lr = 0.1
momentum = 0.9
gradient = 2.0

print("=" * 70)
print(f"{'Step':<6}{'SGD更新':<15}{'Momentum速度':<20}{'Momentum更新':<20}")
print("=" * 70)

# SGD
sgd_update = -lr * gradient

# Momentum
velocity = 0

for step in range(1, 11):

    # 普通 SGD
    sgd = -lr * gradient

    # Momentum SGD
    velocity = momentum * velocity - lr * gradient

    print(f"{step:<6}{sgd:<15.4f}{velocity:<20.4f}{velocity:<20.4f}")