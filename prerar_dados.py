# Imports
import os
import numpy as np
import shutil

''' Preparação dos Dados '''

# Criação das Pastas

original_data_folder = 'data'
class_names = ['cloudy', 'desert', 'green_area', 'water']

# Loop para criar as pastas

for i in range(len(class_names)):

    # Extrair o nome de cada class
    class1 = '/' + class_names[i]


    # Crair as pastas
    os.makedirs('dados/treino/' + class_names[i])
    os.makedirs('dados/val/' + class_names[i])
    os.makedirs('dados/teste/' + class_names[i])


# Cópia das Imagens Para as Pastas

for k in range(len(class_names)):

    # Extrai um nome de classe
    class_name = class_names[k]

    # Define a fonte
    src = original_data_folder + '/' + class_name

    # Mostra qual classe estamos processando
    print("\nClasse:", class_names[k])

    # Lista o conteúdo da pasta
    allFileNames = os.listdir(src)

    # "Embaralha" os dados
    np.random.shuffle(allFileNames)

    # Divisão = 70% treino, 15% teste, 15% validação
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])

    # Nome dos arquivos
    train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
    val_FileNames = [src + '/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]

    # Print
    print('Número Total de Imagens: ', len(allFileNames))
    print('Imagens de Treino: ', len(train_FileNames))
    print('Imagens de Validação: ', len(val_FileNames))
    print('Imagens de Teste: ', len(test_FileNames))

    # Copia as imagens
    for name in train_FileNames:
        shutil.copy(name, "dados/treino/" + class_name)

    for name in val_FileNames:
        shutil.copy(name, "dados/val/" + class_name)

    for name in test_FileNames:
        shutil.copy(name, "dados/teste/" + class_name)
