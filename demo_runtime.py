"""Inferencia da demonstracao 3.9; sem treino ou acesso a benchmarks."""

import contextlib
import hashlib
import html
import io
import json
import math
import shutil
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image, ImageOps, UnidentifiedImageError
from IPython.display import HTML, display


RUN_NAME = 'fase3_9_tgif_sd2fr_v1'
ARCH_SOURCE_SHA256 = '3db23c93de1130c4b55a1b1ad2f3f522005e5e2dd3d15733698f1f2418eb7c7b'
MAX_UPLOAD_BYTES = 20_000_000
MAX_PIXELS = 50_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS // 2  # PIL recusa >50 MP antes de decodificar.


def load_demo_model(run_dir, detector_class, device):
    """Prefere o final calibrado; usa best.pt apenas como demonstracao parcial."""
    run_dir = Path(run_dir)
    complete_path = run_dir / 'training_complete.json'
    final_path = run_dir / 'phase3_final.pt'
    best_path = run_dir / 'best.pt'
    if complete_path.is_file() and final_path.is_file():
        complete = json.loads(complete_path.read_text(encoding='utf-8'))
        if complete.get('status') != 'internal_training_complete':
            raise ValueError('Marcador de treino nao confirma a conclusao')
        source, mode = final_path, 'final_calibrado'
    elif best_path.is_file():
        source, mode = best_path, 'melhor_epoca_sem_calibracao'
    else:
        raise FileNotFoundError('A Fase 3.9 ainda nao tem best.pt nem phase3_final.pt. '
                                'Conclua ao menos uma epoca antes da demonstracao.')

    local = Path('/content/demo_fase3_9_checkpoint.pt')
    shutil.copy2(source, local)
    with local.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    # Carregar apenas o checkpoint confiavel do proprio grupo, nunca um upload.
    checkpoint = torch.load(local, map_location='cpu', weights_only=False)
    if mode == 'final_calibrado':
        required = {'model_state_dict', 'cfg', 'cfg_phase3', 'temperature', 'threshold'}
        if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
            raise ValueError('Checkpoint final incompleto')
        config = checkpoint['cfg_phase3']
        architecture = checkpoint['cfg']
        if (tuple(architecture['BRANCHES']) != ('vit', 'fft', 'cnn') or
                architecture['FUSION'] != 'gate' or architecture['PROJ_DIM'] != 256):
            raise ValueError('Arquitetura do checkpoint inesperada')
        weights = checkpoint['model_state_dict']
        temperature = float(checkpoint['temperature'])
        threshold = float(checkpoint['threshold'])
        epoch = config.get('best_epoch', 'final')
    else:
        if not isinstance(checkpoint, dict) or not {'model', 'cfg', 'epoch'} <= checkpoint.keys():
            raise ValueError('Checkpoint best.pt incompleto')
        config = checkpoint['cfg']
        weights = checkpoint['model']
        temperature, threshold = 1.0, None
        epoch = checkpoint['epoch']
    if config.get('RUN_NAME') != RUN_NAME:
        raise ValueError('Checkpoint nao pertence a Fase 3.9')
    if config.get('SOURCE_SHA256') != ARCH_SOURCE_SHA256:
        raise ValueError('Codigo da arquitetura original difere desta demonstracao')
    if not math.isfinite(temperature) or not 0 < temperature < 10:
        raise ValueError('Temperatura invalida')
    if threshold is not None and (not math.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError('Limiar invalido')

    model = detector_class(branches=('vit', 'fft', 'cnn'), fusion='gate',
                           proj_dim=256, pretrained_cnn=False)
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    return {'model': model, 'mode': mode, 'temperature': temperature,
            'threshold': threshold, 'epoch': epoch, 'checkpoint': str(source),
            'sha256': checksum, 'device': device}


_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
_five_crop = transforms.FiveCrop(224)


def predict_fake_probability(image, loaded):
    """Cinco recortes, media dos logits, temperatura interna quando disponivel."""
    if image.width * image.height > MAX_PIXELS:
        raise ValueError('Imagem acima do limite de 50 megapixels')
    image = ImageOps.exif_transpose(image).convert('RGB')
    if min(image.size) < 224:
        scale = 224 / min(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)),
                             Image.Resampling.BICUBIC)
    batch = torch.stack([_normalize(crop) for crop in _five_crop(image)])
    batch = batch.to(loaded['device'])
    amp = (torch.autocast(device_type='cuda', dtype=torch.float16)
           if loaded['device'].type == 'cuda' else contextlib.nullcontext())
    with torch.inference_mode(), amp:
        logits = loaded['model'](batch).float()
    if logits.shape != (5, 2) or not torch.isfinite(logits).all():
        raise ValueError('Saida do modelo invalida')
    return float(torch.softmax(logits.mean(dim=0) / loaded['temperature'], dim=0)[1])


def show_result(name, image, p_fake, loaded):
    p_real = 1 - p_fake
    threshold = loaded['threshold']
    if threshold is None:
        conclusion = 'Checkpoint ainda sem calibracao; sem decisao binaria.'
        note = 'Pontuacao softmax ilustrativa. A temperatura e o limiar ainda nao foram fixados.'
    else:
        conclusion = ('Mais compativel com imagem sintetica' if p_fake >= threshold
                      else 'Mais compativel com imagem real')
        note = 'Estimativa calibrada internamente; nao comprova a origem da imagem.'
    preview = image.copy()
    preview.thumbnail((500, 400))
    display(preview)
    safe_name = html.escape(name)
    display(HTML(
        '<div style="font:16px Arial;max-width:540px;padding:18px;'
        'border:1px solid #ccc;border-radius:12px">'
        f'<b>{safe_name}</b><p style="font-size:20px">{conclusion}</p>'
        f'<p>Sintetica: <b>{p_fake * 100:.1f}%</b></p>'
        f'<div style="background:#eee;height:18px"><div style="background:#a24de0;'
        f'width:{p_fake * 100:.1f}%;height:18px"></div></div>'
        f'<p>Real: <b>{p_real * 100:.1f}%</b></p>'
        f'<div style="background:#eee;height:18px"><div style="background:#218c78;'
        f'width:{p_real * 100:.1f}%;height:18px"></div></div>'
        f'<p style="font-size:13px">{note}</p></div>'
    ))


def analyze_browser_upload(loaded):
    """Seletor de arquivos do navegador Colab; nao funciona em todo VS Code."""
    from google.colab import files
    uploaded = files.upload()
    for name, raw in uploaded.items():
        try:
            if len(raw) > MAX_UPLOAD_BYTES:
                raise ValueError('Arquivo acima de 20 MB')
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                probability = predict_fake_probability(image, loaded)
                show_result(name, image, probability, loaded)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            print(f'{name}: imagem nao analisada ({error})')


def analyze_drive_folder(loaded, folder):
    """Alternativa ao upload no VS Code: imagens sao lidas da pasta do Drive."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in folder.iterdir() if p.is_file() and
                   p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
    print('Pasta de entrada:', folder, '| imagens:', len(paths))
    for path in paths:
        try:
            if path.stat().st_size > MAX_UPLOAD_BYTES:
                raise ValueError('Arquivo acima de 20 MB')
            with Image.open(path) as image:
                image.load()
                probability = predict_fake_probability(image, loaded)
                show_result(path.name, image, probability, loaded)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            print(f'{path.name}: imagem nao analisada ({error})')
