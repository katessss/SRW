import pandas as pd
import librosa as lr
import librosa.display
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
from sklearn.utils import resample
import seaborn as sns
from sklearn.model_selection import train_test_split
import os, fnmatch
from tqdm import tqdm
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# --- EDA ---
dfA = pd.read_csv('set_a.csv')
dfB = pd.read_csv('set_b.csv')
dfAtiming = pd.read_csv('set_a_timing.csv')

dfA = dfA[dfA.label != 'artifact'] # удалим плохие записи пока что 
dfA = dfA.dropna(subset=['label']) # и записи без меток
dfB = dfB.dropna(subset=['label'])

# Объеденим
dfAB = [dfA,dfB]
dfAB = pd.concat(dfAB)

# немного уровняем данные 

df_murmur = dfAB[dfAB['label'] == 'murmur']
df_extrastole = dfAB[dfAB['label'] == 'extrastole']
df_extrahls = dfAB[dfAB['label'] == 'extrahls']
df_normal = dfAB[dfAB['label'] == 'normal']

RANDOM_SEED = 100
df_exhals_upsampled = resample(df_extrahls, replace=True, n_samples=130, random_state=RANDOM_SEED)  
df_extrastole_upsampled = resample(df_extrastole, replace=True, n_samples=130, random_state=RANDOM_SEED)
df_murmur_upsampled = resample(df_murmur, replace=True, n_samples=130, random_state=RANDOM_SEED)
df_normal_downsampled = resample(df_normal, replace=False, n_samples=100, random_state=RANDOM_SEED)                               

df_upsampled = pd.concat([df_exhals_upsampled, df_extrastole_upsampled])
df_upsampled = pd.concat([df_upsampled, df_normal_downsampled])
ABdf = pd.concat([df_murmur_upsampled, df_upsampled])


ABdf = ABdf.reset_index()

# В фалах аудио называются немного иначе чем в csv 
dir = '/'

audio_list = []
labels = []
for j in range(len(ABdf)):
  if('Btraining_extrastole' in ABdf['fname'][j]):
    alteredPathName = ABdf['fname'][j].replace('Btraining_extrastole', 'extrastole_')
    audio_list.append(dir + alteredPathName)
    labels.append(ABdf["label"][j])
  elif('Btraining_normal_Btraining_' in ABdf['fname'][j]):
    alteredPathName = ABdf['fname'][j].replace('Btraining_normal_Btraining_', 'normal_')
    audio_list.append(dir + alteredPathName)
    labels.append(ABdf["label"][j])
  elif('Btraining_normal' in ABdf['fname'][j]):
    alteredPathName = ABdf['fname'][j].replace('Btraining_normal', 'normal_')
    audio_list.append(dir + alteredPathName)
    labels.append(ABdf["label"][j])
  elif('Btraining_murmur_Btraining_' in ABdf['fname'][j]):
    alteredPathName = ABdf['fname'][j].replace('Btraining_murmur_Btraining_', 'murmur_')
    audio_list.append(dir + alteredPathName)
    labels.append(ABdf["label"][j])
  elif('Btraining_murmur' in ABdf['fname'][j]):
    alteredPathName = ABdf['fname'][j].replace('Btraining_murmur', 'murmur_')
    audio_list.append(dir + alteredPathName)
    labels.append(ABdf["label"][j])
  else:
    audio_list.append(dir + str(ABdf["fname"][j]))
    labels.append(ABdf["label"][j])


audio_df = pd.concat([pd.Series(audio_list, name="audio"), pd.Series(labels, name="label")], axis=1)



# --- Препроцессинг ---
def load_file_data(
    file_names,
    duration=10,
    sr=22050,
    n_mels=64,
    n_fft=2048,
    hop_length=512,
    fmax = 4000
):
    input_length = sr * duration
    data = []

    for file_name in tqdm(file_names):
        # print("load file", file_name)

         # Загружаем аудиофайл с нужной частотой дискретизации
        array, sampling_rate = lr.load(file_name, sr=sr)

        # если короче - дополняем, если длиннее - обрезаем
        if array.shape[0] < input_length:
            # print("fixing audio length:", file_name)
            array = lr.util.fix_length(data=array, size=input_length)
        else:
            array = array[:input_length]

        #Вычисляем мел-спектрограмму, используя переданные параметры
        mel_spectrogram = lr.feature.melspectrogram(
            y=array, 
            sr=sampling_rate, 
            n_mels=n_mels, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        
        # Преобразуем амплитуды в логарифмическую шкалу (децибелы)
        log_mel = lr.power_to_db(mel_spectrogram, ref=np.max)

        data.append(log_mel)

    return data



mapping_dict = {'extrahls': 0, 'extrastole': 1, 'murmur': 2, 'normal': 3}
reversed_dict = {v: k for k, v in mapping_dict.items()}

X = audio_df['audio']
y = audio_df['label'].map(mapping_dict) # переводим в интовые метки 

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f'train {y_train.value_counts()}')

X_train_lms = load_file_data(X_train.values)
X_val_lms = load_file_data(X_val.values)
X_test_lms = load_file_data(X.values)


class HeartDataset(Dataset):
    def __init__ (self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        tensor_X = torch.from_numpy(self.X[ind]).float()
        tensor_y = torch.tensor(self.y[ind], dtype=torch.long)
        return tensor_X, tensor_y
    

train_dataset = HeartDataset(X_train_lms, y_train.values)
val_dataset = HeartDataset(X_val_lms, y_val.values)
test_dataset = HeartDataset(X_test_lms, y.values)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader =  DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader =  DataLoader(test_dataset, batch_size=16, shuffle=False)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # создаём матрицу нулей размером [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # вектор позиций (0, 1, 2, ..., max_len-1), форма [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # делители для частот, экспоненциально уменьшаются
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))

        # применяем синус для чётных индексов признаков
        pe[:, 0::2] = torch.sin(position * div_term)

        # применяем косинус для нечётных индексов признаков
        pe[:, 1::2] = torch.cos(position * div_term)

        # добавляем размерность batch=1 и делаем форму [max_len, 1, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)

        # сохраняем как buffer (не обучаемый параметр, но сохраняется вместе с моделью)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    

class AudioTransformer(nn.Module):
    def __init__(
        self,
        n_mels=64,
        d_model=128,
        nhead=2, # пока что оптимальное
        num_layers=4,
        num_classes=4,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()

        # Линейное преобразование (n_mels => d_model)
        self.input_fc = nn.Linear(n_mels, d_model)

        # Нормализация 
        self.input_norm = nn.LayerNorm(d_model)

        # Позиционное кодирование
        self.pos_encoder = PositionalEncoding(d_model)

        # Трансформер-энкодер
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Финальные блоки
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [T, B, n_mels]
        # return: [batch, num_classes]
        T, B, F = x.shape
        
        x = self.input_norm(self.input_fc(x)) # [T, B, n_mels] => [T, B, d_model] + нормализация по d_model
        x = self.pos_encoder(x)  # Позиционное кодирование
        x = self.transformer_encoder(x)  # (T, B, d_model)
        x = x.mean(dim=0) # (B, d_model) +mean-pool по времени
        x = self.norm(x)
        return self.fc_out(x) # (B, num_classes)
    


def train(train_data, val_data, model, epochs, learning_rate, device):
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    losses, val_losses = [], []
    for epoch in range(epochs):
        epoch_loss, seen = 0, 0
        val_epoch_loss, val_seen = 0, 0
        
        model.train()
        for batch in (bar:=tqdm(train_data)):
            x, y = batch[0].to(device), batch[1].to(device)
            x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True)+1e-6) # центрируем каждый временной фрейм и нормируем по дисперсии
            x = x.permute(2, 0, 1)
            outputs = model(x)

            loss = loss_fn(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            seen += bs
            epoch_loss += loss.item() * bs
            bar.set_description(f'Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}')

        model.eval()
        with torch.no_grad():
            for batch in val_data:
                x, y = batch[0].to(device), batch[1].to(device)
                x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True)+1e-6)
                x = x.permute(2, 0, 1)
                outputs = model(x)     
                loss = loss_fn(outputs, y)
                bs = y.size(0)
                val_seen += bs
                val_epoch_loss += loss.item()*bs
       
        # model.train()
        # scheduler.step()

        epoch_loss /= seen
        val_epoch_loss /= val_seen

        losses.append(epoch_loss)
        val_losses.append(val_epoch_loss)
        print(f'Avg epoch loss {epoch_loss} | Avg val epoch loss {val_epoch_loss}')

    return model, losses, val_losses
            

model = AudioTransformer()
model, losses, val_losses = train(
    train_data=train_loader, 
    val_data=val_loader, 
    model=model, 
    epochs=45, 
    learning_rate=1e-4,
    device='cuda')

trues = []
preds = []
probs = []
device = 'cuda'

model.eval()
with torch.no_grad():
    for batch in test_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True)+1e-6)
        x = x.permute(2, 0, 1)
        logits = model(x)                             

        pred = logits.argmax(dim=1)                  

        preds.extend(pred.cpu().tolist())
        trues.extend(y.cpu().tolist())
        probs.extend(F.softmax(logits, dim=1).cpu().numpy())




report = classification_report(
    y_true=trues, 
    y_pred=preds)
print(report)

cm = confusion_matrix(trues, preds)
disp = ConfusionMatrixDisplay(cm)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap='Blues', ax=ax)
plt.show()


from sklearn.metrics import roc_auc_score
auc = roc_auc_score(trues, probs, multi_class='ovr', average='macro')
print("ROC-AUC:", auc)

#             precision    recall  f1-score   support

#            0       0.99      1.00      1.00       130
#            1       0.70      1.00      0.83       130
#            2       0.93      0.88      0.90       130
#            3       0.90      0.47      0.62       100

#     accuracy                           0.86       490
#    macro avg       0.88      0.84      0.84       490
# weighted avg       0.88      0.86      0.85       490


# ROC-AUC: 0.96