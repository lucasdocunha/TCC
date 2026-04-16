from src.data import ImageDataset
from src.models import xception
from src.plots import *

import logging
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

import numpy as np

import tqdm 
from pathlib import Path
import os

logger = logging.getLogger(__name__)

from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score
from scipy.special import softmax


def run_xception():

    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    logger.info("Iniciando run_xception (pipeline Xception).")

    #configurações básicas para facilitar tudo
    PWD = Path.cwd()
    BATCH = 32
    DEVICE  = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: %s | batch_size=%d | cwd=%s", DEVICE, BATCH, PWD)


    #paralelismo -> caso ache que o computador não aguente 
    # pode deixar tudo como False e deixar só um worker
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    logger.info(
        "DataLoader: num_workers=%d, pin_memory=%s, persistent_workers=%s",
        NUM_WORKERS,
        PIN_MEMORY,
        PERSISTENT_WORKERS,
    )

    #isso é a normalização da imagem, varia de modelo para modelo
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    #crio o dataset de imagens
    logger.info("Carregando ImageDataset (train, val, test)...")

    #espefificações dos dados:
    DATA_LIMIT = np.inf
    FOURIER = "none"

    train = ImageDataset(file_csv=f'{PWD}/data/raw/train.csv', 
        images_dir=f'/media/ssd2/lucas.ocunha/datasets/phase1/trainset', 
        transform=transform,
        data_limit=DATA_LIMIT,
        fourier=FOURIER)

    
    val = ImageDataset(file_csv=f'{PWD}/data/raw/val.csv', 
    images_dir=f'/media/ssd2/lucas.ocunha/datasets/phase1/valset', 
    transform=transform,
    data_limit=DATA_LIMIT,
    fourier=FOURIER)

    test = ImageDataset(file_csv=f'{PWD}/data/raw/test.csv', 
    images_dir=f'/media/ssd2/lucas.ocunha/datasets/phase1/testset', 
    transform=transform,
    data_limit=DATA_LIMIT,
    fourier=FOURIER)

    logger.info(
        "Datasets prontos: amostras train=%s, val=%s, test=%s",
        len(train),
        len(val),
        len(test),
    )

    #transformo eles em um Dataloader -> Classe que vai carregar na memória os dados
    train_loader = DataLoader(train, batch_size=BATCH, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS, pin_memory=PIN_MEMORY, shuffle=False) #shuffle só nos dados de treino e validação
    test_loader = DataLoader(test, batch_size=BATCH, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)
    
    logger.info(
        "DataLoaders criados: batches train=%d, val=%d, test=%d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )


    #Configurações do modelo

    #modelo:

    #pego o primeiro dado do dataset para saber quantos canais de entrada o modelo vai ter
    sample_x, _, _ = train[0]
    in_channels = sample_x.shape[0]
    logger.info(
        "Instanciando Xception pré-treinado e configurando transfer learning... in_channels=%d",
        in_channels,
    )
    model = xception(pretrained=True, in_channels=in_channels) #crio o modelo, pego ele pretreinado já -> aproveitar oq tem
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
    logger.info("Camadas block12, conv3, conv4 e fc liberadas para treino; demais congeladas.")

    # manda o modelo pra GPU/CPU
    model = model.to(DEVICE)
    logger.info("Modelo enviado para %s.", DEVICE)

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

    logger.info(
        "Otimizador Adam (lr fc=1e-3, demais=1e-4), CrossEntropyLoss, AMP GradScaler e ReduceLROnPlateau configurados."
    )

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

    num_epochs = 20
    best_val_loss = float('inf')
    best_path = f"models/{model_name}/weights/best_{model_name}.pth"
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    logger.info("Iniciando treinamento: %d épocas. Melhor modelo será salvo em %s", num_epochs, best_path)

    #tqdm só para deixar bonitinho
    epoch_bar = tqdm.tqdm(range(num_epochs), desc="Epochs")
    for epoch in epoch_bar:
        
        #treino:
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        train_pbar = tqdm.tqdm(
            train_loader,
            desc=f"Train ep {epoch + 1}/{num_epochs}",
            leave=False,
        )
        for img, label, idx in train_pbar:
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
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        #validação:
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            val_pbar = tqdm.tqdm(
                val_loader,
                desc=f"Val ep {epoch + 1}/{num_epochs}",
                leave=False,
            )
            for x, y, _ in val_pbar:
                x = x.to(DEVICE)
                y = y.to(DEVICE)

                out = model(x)
                loss = criterion(out, y)

                val_loss += loss.item()
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            logger.info(
                "Época %d/%d: novo melhor val_loss=%.6f — checkpoint salvo em %s",
                epoch + 1,
                num_epochs,
                val_loss,
                best_path,
            )

        #faço o step do scheduler ser só conforme a validação
        #precisa não melhorar a %de acerto na validação, para ai sim ele atualizar
        scheduler.step(val_loss)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
        )
        logger.info(
            "Época %d/%d concluída | train_loss=%.6f train_acc=%.6f | val_loss=%.6f val_acc=%.6f",
            epoch + 1,
            num_epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

    #teste:
    logger.info("Treino finalizado. Carregando melhores pesos de %s para avaliação no teste.", best_path)
    test_results = {}
    y_true, y_pred = [], []
    all_logits, all_ids = [], []

    model.load_state_dict(torch.load(best_path))
    model.eval()

    with torch.no_grad():
        for x, y, idx in tqdm.tqdm(test_loader, desc="Teste (inferência)"):
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

    logger.info(
        "Métricas no teste: acc=%.6f precision=%.6f f1=%.6f auc=%.6f",
        test_results["acc"],
        test_results["precision"],
        test_results["f1"],
        test_results["auc"],
    )
        
    #agora vou salvar os resultados aqui:
    model_dir = f'{PWD}/models/{model_name}/{FOURIER}'

    os.makedirs(os.path.join(model_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'results'), exist_ok=True)
    logger.info("Salvando artefatos em %s (weights + results)...", model_dir)

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
    logger.info("Pesos finais salvos em models/%s/weights/%s.pth", model_name, model_name)

    logger.info("Gerando gráficos (matriz de confusão e ROC-AUC)...")
    plot_confusion_matrix(test_results, model_dir, f'{model_name} Confusion Matrix')
    plot_roc_auc(test_results, model_dir, f'{model_name} ROC-AUC Curve')

    extra_info = {
        'Camadas descongeladas': 3,
        'Nº épocas': num_epochs
    }

    save_metrics_csv(test_results, model_dir, extra_info=extra_info)
    logger.info("run_xception concluído (métricas exportadas e plots salvos).")