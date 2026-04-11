from src.data import ImageDataset
from src.models import xception
from src.plots import *

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import numpy as np

import tqdm 
from pathlib import Path
import os

from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
from scipy.special import softmax


def run_xception():

    #configurações básicas para facilitar tudo
    PWD = Path.cwd()
    BATCH = 32
    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #paralelismo -> caso ache que o computador não aguente 
    # pode deixar tudo como False e deixar só um worker
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True


    #crio o dataset de imagens
    train = ImageDataset(file_csv=f'{PWD}/data/raw/train.csv', images_dir=f'/media/ssd2/lucas.ocunha/datasets/phase1/trainset')
    val = ImageDataset(file_csv=f'{PWD}/data/raw/val.csv', images_dir=f'/media/ssd2/lucas.ocunha/datasets/phase1/valset')
    test = ImageDataset(file_csv=f'{PWD}/data/raw/test.csv', images_dir=f'/media/ssd2/lucas.ocunha/datasets/phase1/testset')

    #transformo eles em um Dataloader -> Classe que vai carregar na memória os dados
    train_loader = DataLoader(train, batch_size=BATCH, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS, pin_memory=PIN_MEMORY, shuffle=False) #shuffle só nos dados de treino e validação
    test_loader = DataLoader(test, batch_size=BATCH, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)


    #Configurações do modelo

    #modelo:
    model = xception(pretrained=True) #crio o modelo, pego ele pretreinado já -> aproveitar oq tem
    model_name = xception.__name__
    #aqui, quero fazer transfer learning (congelar as primeiras camadas do meu modelo e treinar somente as últimas camadas + parte do classificador)
    # o modelo pode ser dividido em duas partes:

    #1. Extrator de características:
    #aqui vou congelar tudo primeiro
    for param in model.parameters():
        param.requires_grad = False
        
    # libera últimas camadas
    for param in model.block12.parameters():
        param.requires_grad = True

    for param in model.conv3.parameters():
        param.requires_grad = True

    for param in model.conv4.parameters():
        param.requires_grad = True
        
    #2. meu classicador -> coloco ele com 2 de saída, pelo número das classes
    model.fc = nn.Linear(2048, 2)

    # manda o modelo pra GPU/CPU
    model = model.to(DEVICE)

    #Aqui estou definindo as coisas de como o modelo vai ser treinado:
    #loss
    criterion = nn.CrossEntropyLoss()

    #Otimizador do modelo -> tenho que passar só as camadas que preciso treinar
    # se não pode dar vazamento de memória e demorar mais tempo

    #learning rates diferentes para ser mais estável o Fine-Tunning
    #1e-3 no fc pq é a parte do classificador (camada nova) -> precisa aprender rapidamente
    #1e-4 nas camadas da rede pq é a parte que já vem pretreinada, mas ela precisa se adaptar as nossas imagens
    optimizer = torch.optim.Adam([
        {'params': model.fc.parameters(), 'lr': 1e-3}, 
        {'params': model.block12.parameters(), 'lr': 1e-4},
        {'params': model.conv3.parameters(), 'lr': 1e-4},
        {'params': model.conv4.parameters(), 'lr': 1e-4},
    ])

    #O scaler ajuda a loss a se manter estável
    scaler = torch.cuda.amp.GradScaler()

    # scheduler (opcional, mas útil)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, #muda o optimizer
        mode='min', #quando acha o minimo
        factor=0.1, #muda 0.1 do learning rate
        patience=4 # tem uma paciencia de duas épocas se não melhorar -> mudar conforme a quantidade de epocas
    )
    #no scheduler, ao usar ele, ele identifica quando o modelo está aprendendo e quando está parando de aprender
    # vai ajustando a taxa de learning rate conforme isso, ajuda que o modelo a convergir

    num_epochs = 10
    #tqdm só para deixar bonitinho
    for epoch in tqdm.tqdm(range(num_epochs),desc=f"Epochs"):
        
        best_val_loss = float('inf')
        best_path = f"models/{model_name}/weights/best_{model_name}.pth"
        
        #treino:
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for img, label, idx in train_loader:
            #mando meus dados para o device
            x = img.to(DEVICE) 
            y = label.to(DEVICE)
            
            
            #faço a otimização no modelo
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                #faço a previsão no meu dado
                out = model(x)
                
                #comparo ele com o real
                loss = criterion(out, y)

            #atualiza a loss, fazendo o backward do modelo (retropropagação)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += y.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        #validação:
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                out = model(x)
                loss = criterion(out, y)

                val_loss += loss.item()
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        #faço o step do scheduler ser só conforme a validação
        #precisa não melhorar a %de acerto na validação, para ai sim ele atualizar
        scheduler.step(val_loss)

    #teste:
    test_results = {}
    y_true, y_pred = [], []
    all_logits, all_ids = [], []

    model.load_state_dict(torch.load(best_path))
    model.eval()

    with torch.no_grad():
        for x, y, idx in tqdm(test_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            with torch.cuda.amp.autocast():
                out = model(x)

            preds = out.argmax(1)
            
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # salva logits -> valores antes do softmax ex: [-1.2, 2.3]
            #salvo eles pq depois, caso precise fazer qualquer coisa, posso recalcular com ele
            all_logits.append(
                out.detach().cpu().numpy().astype(np.float32)
            )

            all_ids.extend(idx.cpu().numpy())
            
    #transformo o logits em um array np
    logits = np.concatenate(all_logits)
    probs = softmax(logits, axis=1)[:, 1]

    #faço todas as métricas
    test_results['acc'] = accuracy_score(y_true, y_pred)
    test_results['precision'] = precision_score(y_true, y_pred)
    test_results['f1'] = f1_score(y_true, y_pred)
    test_results['auc'] = roc_auc_score(y_true, probs)
    test_results['y_true'] = y_true
    test_results['y_pred'] = y_pred
    test_results['logits'] = logits
    test_results['ids'] = np.array(all_ids)
        
    #agora vou salvar os resultados aqui:
    model_dir = f'{PWD}/models/{model_name}'

    os.makedirs(os.path.join(model_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'results'), exist_ok=True)

    #salvo o arquivo numpy para não precisar rodar novamente
    np.savez_compressed(
            f"{model_dir}/results/outputs.npz",
            logits=test_results["logits"],   
            ids=test_results["ids"]
        )

    #salvo o modelo treinado
    torch.save(
                model.state_dict(),
                f"models/{model_name}/weights/{model_name}.pth"
    )

    plot_confusion_matrix(test_results, model_dir, f'{model_name} Confusion Matrix')
    plot_roc_auc(test_results, model_dir, f'{model_name} ROC-AUC Curve')

    extra_info = {
        'Camadas descongeladas': 3,
        'Nº épocas': num_epochs
    }

    save_metrics_csv(test_results, model_dir, extra_info=extra_info)