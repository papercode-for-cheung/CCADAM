import pandas as pd
import os

# 读取CSV文件
df = pd.read_csv(os.path.join('.','OS','DDinter2.0.csv'))

# 分离label为0和1的数据
df_label_0 = df[df['label'] == 0]
df_label_1 = df[df['label'] == 1]

# 计算总数据量
total_samples = len(df_label_0)

# 计算训练集、验证集和测试集的大小
train_size = int(total_samples * 0.7)
val_size = int(total_samples * 0.2)
test_size = total_samples - train_size - val_size

# 确保每个数据集中的label为0和1的数量是1:1
train_size_0 = train_size 
train_size_1 = train_size

val_size_0 = val_size 
val_size_1 = val_size 

test_size_0 = test_size 
test_size_1 = test_size 


# 随机抽取数据
train_0 = df_label_0.sample(n=train_size_0, random_state=42)
train_1 = df_label_1.sample(n=train_size_1, random_state=42)

val_0 = df_label_0.drop(train_0.index).sample(n=val_size_0, random_state=42)
val_1 = df_label_1.drop(train_1.index).sample(n=val_size_1, random_state=42)

test_0 = df_label_0.drop(train_0.index).drop(val_0.index).sample(n=test_size_0, random_state=42)
test_1 = df_label_1.drop(train_1.index).drop(val_1.index).sample(n=test_size_1, random_state=42)

# 合并数据
train_set = pd.concat([train_0, train_1])
val_set = pd.concat([val_0, val_1])
test_set = pd.concat([test_0, test_1])

train_set = train_set.sample(frac=1, random_state=42).reset_index(drop=True)
val_set = val_set.sample(frac=1, random_state=42).reset_index(drop=True)
test_set = test_set.sample(frac=1, random_state=42).reset_index(drop=True)

train_set.to_csv(os.path.join('.','OS','DDinter2.0_train.csv'), index=False)
val_set.to_csv(os.path.join('.','OS','DDinter2.0_valid.csv'), index=False)
test_set.to_csv(os.path.join('.','OS','DDinter2.0_test.csv'), index=False)

print("Train set label distribution:")
print(train_set['label'].value_counts())

print("Validation set label distribution:")
print(val_set['label'].value_counts())

print("Test set label distribution:")
print(test_set['label'].value_counts())