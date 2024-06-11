from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn
import torch
from collections import OrderedDict
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import os
import zipfile
import shutil
from PIL import Image
import random
import matplotlib.pyplot as plt
import json
import torchvision.transforms as transforms

app = FastAPI()

# Definir los nombres de las clases
class_names = ["No-Anomaly", "Offline-Module", "Cell", "Vegetation", "Diode-Multi", "Diode", 
               "Cell-Multi", "Shadowing", "Cracking", "Hot-Spot", "Hot-Spot-Multi", "Soiling"]

# Ruta del modelo guardado
model_path = 'model.pth'

# Función para cargar y ajustar el modelo
def load_adjusted_model(model_path, class_names, device):
    # Crear el modelo preentrenado
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_ftrs = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(class_names))
    )
    
    # Cargar el estado del modelo guardado
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k == "_fc.weight":
            new_state_dict["_fc.1.weight"] = v
        elif k == "_fc.bias":
            new_state_dict["_fc.1.bias"] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    return model

# Seleccionar dispositivo (GPU si está disponible, de lo contrario CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo ajustado
model = load_adjusted_model(model_path, class_names, device)
model = model.to(device)
model.eval()

# Definir la transformación de la imagen
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Función para preprocesar la imagen, hacer la predicción y verificar
def predict_and_verify_image(image_path, model, class_names, json_path, device):
    # Transformación de la imagen
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Cargar y transformar la imagen
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Hacer la predicción
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
    
    predicted_class = class_names[preds.item()]

    # Obtener la clase real desde el archivo JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    image_name = os.path.basename(image_path)
    actual_class = None
    for key, value in data.items():
        if os.path.basename(value["image_filepath"]) == image_name:
            actual_class = value["anomaly_class"]
            break
    
    return predicted_class, actual_class

# Función para realizar predicciones aleatorias
def random_predictions(image_folder, model, class_names, json_path, device, num_images=20):
    # Obtener una lista de todas las imágenes en la carpeta
    all_images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Seleccionar un subconjunto aleatorio de imágenes
    selected_images = random.sample(all_images, min(num_images, len(all_images)))
    
    correct_predictions = 0
    incorrect_predictions = 0
    results = []
    class_correct = {cls: 0 for cls in class_names}
    class_incorrect = {cls: 0 for cls in class_names}

    # Realizar predicciones y verificar cada imagen seleccionada
    for image_path in selected_images:
        predicted_class, actual_class = predict_and_verify_image(image_path, model, class_names, json_path, device)
        if predicted_class == actual_class:
            correct_predictions += 1
            class_correct[actual_class] += 1
        else:
            incorrect_predictions += 1
            class_incorrect[actual_class] += 1
        results.append({
            "image_path": image_path,
            "predicted_class": predicted_class,
            "actual_class": actual_class,
            "correct": predicted_class == actual_class
        })

    # Calcular la efectividad
    effectiveness = correct_predictions / len(results) * 100

    return {
        "total_images": len(results),
        "correct_predictions": correct_predictions,
        "incorrect_predictions": incorrect_predictions,
        "effectiveness": effectiveness,
        "results": results,
        "class_correct": class_correct,
        "class_incorrect": class_incorrect
    }

# Función para generar y guardar el gráfico de barras
def generate_bar_chart(data, file_path):
    class_names = list(data['class_correct'].keys())
    correct_counts = list(data['class_correct'].values())
    incorrect_counts = list(data['class_incorrect'].values())
    total_counts = [correct + incorrect for correct, incorrect in zip(correct_counts, incorrect_counts)]
    effectiveness = [correct / total * 100 if total != 0 else 0 for correct, total in zip(correct_counts, total_counts)]

    x = range(len(class_names))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, correct_counts, width=0.4, label='Correctas', align='center')
    ax.bar(x, incorrect_counts, width=0.4, bottom=correct_counts, label='Incorrectas', align='center')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('Número de Predicciones')
    ax.set_title('Predicciones Correctas e Incorrectas por Clase')
    ax.legend()

    for i, (eff, correct, total) in enumerate(zip(effectiveness, correct_counts, total_counts)):
        ax.text(i, total + 0.5, f'{eff:.2f}%', ha='center')

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

@app.post("/upload-zip/")
async def upload_zip(file: UploadFile = File(...)):
    zip_path = 'images_filtered.zip'
    extract_path = 'images_filtered/'

    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print("Extracción completada.")
    return {"message": "Extracción completada"}

@app.get("/predict-random/{num_images}")
async def predict_random(num_images: int):
    image_folder = 'images_filtered/'  # Cambia esto a la ruta de tu carpeta de imágenes
    json_path = 'updated_module_metadata.json'  # Cambia esto a la ruta de tu archivo JSON

    results = random_predictions(image_folder, model, class_names, json_path, device, num_images=num_images)
    generate_bar_chart(results, 'prediction_chart.png')
    return results

@app.get("/chart")
async def get_chart():
    file_path = 'prediction_chart.png'
    return FileResponse(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



