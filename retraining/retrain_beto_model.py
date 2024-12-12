import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments

# Cargar el dataset
dataset_path = "data/retrain/dataset_terminos.csv"  # Ruta a tu dataset
dataset = pd.read_csv(dataset_path)

# Definir un Dataset personalizado para PyTorch
class TranscriptionCorrectionDataset(Dataset):
    def __init__(self, transcriptions, corrections):
        self.transcriptions = transcriptions
        self.corrections = corrections
        self.tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

    def __len__(self):
        return len(self.transcriptions)

    def __getitem__(self, idx):
        # Crear una entrada con una palabra enmascarada
        text = self.transcriptions[idx].replace("[MASK]", self.corrections[idx])
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs["labels"] = inputs["input_ids"].clone()  # Las etiquetas son las mismas que los inputs
        return {k: v.squeeze() for k, v in inputs.items()}

# Preparar el dataset para el entrenamiento
transcriptions = dataset["transcripcion_erronea"].tolist()
corrections = dataset["correccion_deseada"].tolist()
train_dataset = TranscriptionCorrectionDataset(transcriptions, corrections)

# Configurar los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./models/beto_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    logging_steps=10
)

# Cargar el modelo BETO para Masked Language Modeling
model = BertForMaskedLM.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

# Crear el objeto Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Reentrenar el modelo
print("Iniciando el reentrenamiento...")
trainer.train()
print("Reentrenamiento completado.")

# Guardar el modelo ajustado
print("Guardando el modelo reentrenado...")
model.save_pretrained("./models/beto_finetuned")
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
tokenizer.save_pretrained("./models/beto_finetuned")
print("Modelo guardado exitosamente.")
