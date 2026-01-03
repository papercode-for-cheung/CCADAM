import pandas as pd
import os
import random

# 读取CSV文件
df = pd.read_csv(os.path.join('.', 'FNOS', 'DDinter2.0.csv'))

# 打乱DataFrame的顺序
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# 全集中Drug A和Drug B的ID
drug_a_ids = set(df_shuffled['Drug A ID'])
drug_b_ids = set(df_shuffled['Drug B ID'])
all_ids = drug_a_ids.union(drug_b_ids)

# 随机划分 all_ids
random.seed(42)  # 设置随机种子以确保结果可重复
all_ids_list = list(all_ids)
random.shuffle(all_ids_list)

# 计算总数据量
total_samples = len(all_ids_list)
train_size = int(total_samples * 0.6)
val_size = int(total_samples * 0.2)
test_size = total_samples - train_size -val_size

train_id = set(all_ids_list[:train_size])
val_id = set(all_ids_list[train_size:train_size + val_size])
test_id = set(all_ids_list[train_size + val_size:])


val = df_shuffled[(df_shuffled['Drug A ID'].isin(val_id)) & (df_shuffled['Drug B ID'].isin(val_id))]

test = df_shuffled[(df_shuffled['Drug A ID'].isin(test_id)) & (df_shuffled['Drug B ID'].isin(test_id))]

train = df_shuffled[(df_shuffled['Drug A ID'].isin(train_id)) & (df_shuffled['Drug B ID'].isin(train_id))]

# combined_index = set(test.index).union(set(val.index))

# # 去除测试集和验证集，得到训练集
# train = df_shuffled.drop(combined_index)

# 确保train、val、test三者的大小从大到小排列
if len(train) < len(val):
    train, val = val, train
if len(train) < len(test):
    train, test = test, train
if len(val) < len(test):
    val, test = test, val

train.to_csv(os.path.join('.','FNOS','DDinter2.0_train.csv'), index=False)
val.to_csv(os.path.join('.','FNOS','DDinter2.0_valid.csv'), index=False)
test.to_csv(os.path.join('.','FNOS','DDinter2.0_test.csv'), index=False)

# 打印三者大小
print(f"Train size: {len(train)}")
print(f"Validation size: {len(val)}")
print(f"Test size: {len(test)}")

print("Train set label distribution:")
print(train['label'].value_counts())

print("Validation set label distribution:")
print(val['label'].value_counts())

print("Test set label distribution:")
print(test['label'].value_counts())


# 6. 输出划分结果和验证信息

print(f"\n验证集和测试集重复情况: {len(pd.merge(val, test, on=['Drug A ID', 'Drug B ID', 'label'], how='inner'))}")