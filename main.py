# Imports
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models
import gc
import types
import pkg_resources
import pytorch_lightning as pl
from sklearn import metrics

if __name__ == '__main__':

    ''' Verificando o Ambiente de Desenvolvimento '''

    # Verifica se uma GPU está disponível e define o dispositivo apropriado

    processing_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define o device (GPU ou CPU)
    device = torch.device(processing_device)
    print(device)

    ''' Transformações '''

    # Define os hiperparâmetros
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.0003

    # Define as Transformações
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Resize((100, 100)),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    ''' Divisão dos Dados '''

    dataset_treino = datasets.ImageFolder('dados/treino', transform=transform)
    dataset_valid = datasets.ImageFolder('dados/val', transform=transform)
    dataset_teste = datasets.ImageFolder('dados/teste', transform=transform)

    print(f"Número Total de Imagens de Treino: {len(dataset_treino)}")
    print(f"Número Total de Imagens de Validação: {len(dataset_valid)}")
    print(f"Número Total de Imagens de Teste: {len(dataset_teste)}")

    # Dataloader de treino
    dataloader_treino = DataLoader(dataset_treino, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
    dataloader_teste = DataLoader(dataset_teste, batch_size=batch_size, shuffle=False)

    ''' Visualização de distribuição das classes de imagens '''

    # Nomes das classes
    class_names = os.listdir('dados/treino')
    print(class_names)

    # Loop para contar as imagens em cada classe
    image_count = {}
    for i in class_names:
        image_count[i] = len(os.listdir(os.path.join('dados/treino', i))) - 1

    print(image_count)

    ''' Modelagem '''

    # Baixamos o modelo pré-treinado de arquitetura DenseNet, incluindo os pesos
    modelo = models.densenet121(weights='DEFAULT')

    # Carregamos os parâmetros (pesos) do modelo
    modelo.parameters()

    # "Congelamos" os pesos do modelo
    for param in modelo.parameters():
        param.requires_grad = False

    # Arquitetura do classificador ("cabeça" do modelo) com ReLu
    clf = nn.Sequential(nn.Linear(1024, 460),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(460, 5))

    # Adiciona a camada de classificação ao modelo pré-treinado (vamos treinar somente a última camada)
    modelo.classifier = clf

    # Envia o modelo para a memória do device
    modelo.to(device)

    # Função de perda
    criterion = nn.CrossEntropyLoss()

    # Otimizador
    optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)

    # Vamos medir o tempo de execução
    total_step = len(dataloader_treino)
    print_every = len(dataloader_treino) - 1
    loss_values = []
    total_step = len(dataloader_treino)
    epoch_times = []

    # Loop de treinamento pelo número de épocas
    for epoch in range(num_epochs):

        # Loop por cada batch de imagem/label
        for i, (images, labels) in enumerate(dataloader_treino):

            # Zera o contador de erro
            running_loss = 0.0

            # Carrega imagens do batch
            images = images.to(device)

            # Carrega labels do batch
            labels = labels.to(device)

            # Faz a previsão com o modelo
            outputs = modelo(images)

            # Calcula o erro do modelo
            loss = criterion(outputs, labels)

            # Zera os gradientes que serão aprendidos
            optimizer.zero_grad()

            # Aplica o backpropagation
            loss.backward()

            # Aplica a otimização dos gradientes (aqui ocorre o aprendizado)
            optimizer.step()

            # Registra o erro do modelo
            running_loss += loss.item()

            # Imprime em intervalos regulares
            if (i + 1) % print_every == 0:
                loss_values.append(running_loss / print_every)

                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}: Batch Loss : {}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), running_loss / print_every))

                running_loss = 0

    # Cria os arrays que vão receber as previsões
    arr_pred = np.empty((0, len(dataset_teste)), int)
    arr_label = np.empty((0, len(dataset_teste)), int)

    # Loop de previsões
    with torch.no_grad():

        # Contadores
        correct = 0
        total = 0

        # Loop pelos dados de teste
        for images, labels in dataloader_teste:
            # Extrai imagens e labels do batch de teste
            images = images.to(device)
            labels = labels.to(device)

            # Previsão com o modelo
            outputs = modelo(images)

            # Extrai o maior valor de probabilidade (classe prevista)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred = predicted.cpu().numpy()
            lb = labels.cpu().numpy()
            arr_pred = np.append(arr_pred, pred)
            arr_label = np.append(arr_label, lb)

        print('Acurácia do Modelo em ' + str(len(dataset_teste)) + ' imagens de teste: {} %'.format(
            100 * correct / total))

    # Salvando o modelo
    torch.save(modelo, 'modelo.pt')