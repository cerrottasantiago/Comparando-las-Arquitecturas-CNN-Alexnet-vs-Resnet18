import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchsummary import summary
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


# %% CALCULO MANUAL DE PESOS DE LA RESNET 18
# - - - Ingreso de las imagenes - - - 
Entrada = 64 * 3 * 7 * 7  # = 9408
Batch0 = 64 * 2

# Bloque 1
Conv1 = 64 * 64 * 3 * 3   # = 36864
Batch1 = 64 * 2
Conv2 = 64 * 64 * 3 * 3
Batch2 = 64 * 2
Conv3 = 64 * 64 * 3 * 3
Batch3 = 64 * 2 
Conv4 = 64 * 64 * 3 * 3
Batch4 = 64 * 2

#-------
# Bloque 2
Conv5 = 128 * 64 * 3 * 3  # = 73728
Batch5 = 128 * 2
Conv6 = 128 * 128 * 3 * 3
Batch6 = 128 * 2
ConvSkip1 = 128 * 64 * 1 * 1  # filtros 1x1
BatchSkip1 = 128 * 2

Conv7 = 128 * 128 * 3 * 3
Batch7 = 128 * 2
Conv8 = 128 * 128 * 3 * 3
Batch8 = 128 * 2

# Bloque 3
Conv9 = 256 * 128 * 3 * 3
Batch9 = 256 * 2
Conv10 = 256 * 256 * 3 * 3
Batch10 = 256 * 2
ConvSkip2 = 256 * 128 * 1 * 1
BatchSkip2 = 256 * 2
Conv11 = 256 * 256 * 3 * 3
Batch11 = 256 * 2
Conv12 = 256 * 256 * 3 * 3
Batch12 = 256 * 2

# Bloque 4
Conv13 = 512 * 256 * 3 * 3
Batch13 = 512 * 2
Conv14 = 512 * 512 * 3 * 3
Batch14 = 512 * 2
ConvSkip3 = 512 * 256 * 1 * 1
BatchSkip3 = 512 * 2
Conv15 = 512 * 512 * 3 * 3
Batch15 = 512 * 2
Conv16 = 512 * 512 * 3 * 3
Batch16 = 512 * 2

# Fully Connected
FC = 512 * 1000 + 1000   # = 512000 + 1000 (Softmax)
 
Resultado = (
    Entrada + Batch0 +
    Conv1 + Batch1 + Conv2 + Batch2 + Conv3 + Batch3 + Conv4 + Batch4 +
    Conv5 + Batch5 + Conv6 + Batch6 + ConvSkip1 + BatchSkip1 + Conv7 + Batch7 + Conv8 + Batch8 +
    Conv9 + Batch9 + Conv10 + Batch10 + ConvSkip2 + BatchSkip2 + Conv11 + Batch11 + Conv12 + Batch12 +
    Conv13 + Batch13 + Conv14 + Batch14 + ConvSkip3 + BatchSkip3 + Conv15 + Batch15 + Conv16 + Batch16
)
print(f"\nResultado sin FC: {Resultado}") # = 11176512
ResultadoConFC = Resultado + FC
print(f"\nResultado con FC: {ResultadoConFC}") # = 11689512

#%%

image_path = r"" # <-- Entre las comillas se tiene que poner la ruta exacta de la imagen (No importa el tamaño ya que luego se aplica un resize).

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True).to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img_pil = Image.open(image_path).convert('RGB')
x = transform(img_pil).unsqueeze(0).to(device)

canal = 29 # <-- Se puede poner cualquier canal si está dentro del límite de canales que tiene el feature map.

out = model.conv1(x)
out = model.bn1(out)
out = model.relu(out)  
FM1 = model.maxpool(out)

# Se puede cambiar el número del layer que representa el sequential.
# Y dentro de los layers con los corchetes se pueden cambiar el número para acceder a distintos bloques.

out = model.layer1[0].conv1(FM1)  
out = model.layer1[0].bn1(out)    
out = model.layer1[0].relu(out)   
out = model.layer1[0].conv2(out) 
FM2 = model.layer1[0].bn2(out) 


suma_manual_calculo = FM1 + FM2
add_resnet_calculo = model.layer1[0](FM1)
suma_manual_con_relu = torch.relu(suma_manual_calculo)

A = FM1[0, canal].cpu().detach().numpy()
B = FM2[0, canal].cpu().detach().numpy()
suma_manual = suma_manual_con_relu[0, canal].cpu().detach().numpy()
add_resnet = add_resnet_calculo[0, canal].cpu().detach().numpy()

plt.figure(figsize=(20, 5))

plt.subplot(1, 5, 1)
sns.heatmap(A, annot=True, fmt='.1f', cmap='viridis', cbar=False,
            annot_kws={"size": 3}, square=True)
plt.title(f'FM1 - Canal {canal}')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.text(0.5, 0.5, '+', fontsize=40, ha='center', va='center')
plt.axis('off')

plt.subplot(1, 5, 3)
sns.heatmap(B, annot=True, fmt='.1f', cmap='viridis', cbar=False,
            annot_kws={"size": 3}, square=True)
plt.title(f'FM2 - Canal {canal}')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.text(0.5, 0.5, '=', fontsize=40, ha='center', va='center')
plt.axis('off')

plt.subplot(1, 5, 5)
sns.heatmap(suma_manual, annot=True, fmt='.1f', cmap='viridis', cbar=False,
            annot_kws={"size": 3}, square=True)
plt.title(f'Suma Manual\nCanal {canal}')
plt.axis('off')

plt.suptitle(f'SUMA MANUAL: FM1 + FM2 (Canal {canal})')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(suma_manual, annot=True, fmt='.1f', cmap='viridis', cbar=False,
            annot_kws={"size": 3}, square=True)
plt.title(f'Suma Manual\nCanal {canal}')
plt.axis('off')

plt.subplot(1, 2, 2)
sns.heatmap(add_resnet, annot=True, fmt='.1f', cmap='viridis', cbar=False,
            annot_kws={"size": 3}, square=True)
plt.title(f'Add ResNet\nCanal {canal}')
plt.axis('off')

plt.tight_layout()
plt.show()

print("¿Suma manual + ReLU es igual al Add natural?:")
suma_manual = suma_manual_con_relu.cpu()
add_resnet = add_resnet_calculo.cpu()
if torch.allclose(suma_manual, add_resnet, rtol=1e-5, atol=1e-8):
    print(" Son iguales")
#%%

summary(model, (3, 224, 224))


print(model)
