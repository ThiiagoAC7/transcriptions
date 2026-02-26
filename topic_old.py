from __future__ import annotations

import argparse
import gc
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import emoji
import numpy as np
import pandas as pd
from bertopic import BERTopic
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP

# from hdbscan import HDBSCAN
# from umap import UMAP

from sentence_transformers import SentenceTransformer


@dataclass
class VideoDocument:
    youtuber: str
    video_id: str
    source: str  # "transcript" ou "comment"
    text: str
    # None para transcript, 0,1,2... para comments
    comment_index: Optional[int] = None


@dataclass
class CleaningConfig:
    enabled: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    process_hashtags: bool = True
    remove_numbers: bool = True
    remove_punctuation: bool = True
    normalize_elongated: bool = True
    expand_slang: bool = True
    remove_stopwords: bool = True
    min_tokens: int = 3


STOPWORDS = {
    # português
    "a", "o", "os", "as", "um", "uma", "de", "da", "do", "das", "dos",
    "e", "é", "em", "no", "na", "nos", "nas", "por", "pra", "para",
    "que", "se", "com", "como", "ao", "aos", "à", "às",
    "eu", "tu", "ele", "ela", "nós", "vos", "eles", "elas", "você", "vocês",
    "me", "te", "lhe", "nos", "lhes",
    "isso", "isto", "aquilo", "aqui", "ali", "lá",
    # inglês
    "the", "a", "an", "and", "or", "but", "if", "then", "else",
    "in", "on", "at", "for", "to", "of", "from", "by", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "them", "my", "your", "his", "their",
}

SLANG_MAP = {
    # inglês
    "u": "you",
    "ur": "your",
    "btw": "by the way",
    "omg": "oh my god",
    "idk": "i do not know",
    "lol": "laugh",
    "wtf": "what the hell",
    # português
    "vc": "você",
    "vcs": "vocês",
    "tb": "também",
    "pq": "porque",
    "q": "que",
    "blz": "beleza",
    "vlw": "valeu",
    "obg": "obrigado",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NUMERIC_RE = re.compile(r"\b\d+\b")
REPEATED_CHARS_RE = re.compile(r"(.)\1{2,}", re.UNICODE)
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def normalize_whitespace_and_case(text: str) -> str:
    text = text.lower()
    text = " ".join(text.split())
    return text


def replace_emojis_with_text(text: str) -> str:
    # emoji.demojize transforma 😀 -> :grinning_face:, que vira token textual
    return emoji.demojize(text, language="en")


def remove_noise_tokens(text: str, config: CleaningConfig) -> str:
    if config.remove_urls:
        text = URL_RE.sub(" ", text)
    if config.remove_mentions:
        text = MENTION_RE.sub(" ", text)
    if config.process_hashtags:
        text = HASHTAG_RE.sub(r"\1", text)
    if config.remove_numbers:
        text = NUMERIC_RE.sub(" ", text)
    if config.remove_punctuation:
        table = str.maketrans({ch: " " for ch in string.punctuation})
        text = text.translate(table)
    text = " ".join(text.split())
    return text


def normalize_repeated_chars(tokens: List[str], config: CleaningConfig) -> List[str]:
    if not config.normalize_elongated:
        return tokens
    normalized = []
    for tok in tokens:
        tok_norm = REPEATED_CHARS_RE.sub(r"\1\1", tok)
        normalized.append(tok_norm)
    return normalized


def expand_slang_and_acronyms(tokens: List[str], config: CleaningConfig) -> List[str]:
    if not config.expand_slang:
        return tokens
    expanded: List[str] = []
    for tok in tokens:
        repl = SLANG_MAP.get(tok, tok)
        expanded.extend(repl.split())
    return expanded


def simple_lemmatize(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def lemmatize_and_remove_stopwords(tokens: List[str], config: CleaningConfig) -> List[str]:
    result: List[str] = []
    for tok in tokens:
        lemma = simple_lemmatize(tok)
        if config.remove_stopwords and lemma in STOPWORDS:
            continue
        if not lemma:
            continue
        result.append(lemma)
    return result


def clean_text(text: str, config: CleaningConfig) -> str:
    if not config.enabled:
        return text

    text = normalize_whitespace_and_case(text)
    text = replace_emojis_with_text(text)
    text = remove_noise_tokens(text, config)

    tokens = TOKEN_RE.findall(text)
    if not tokens:
        return ""

    tokens = normalize_repeated_chars(tokens, config)
    tokens = expand_slang_and_acronyms(tokens, config)
    tokens = lemmatize_and_remove_stopwords(tokens, config)

    if not tokens:
        return ""

    return " ".join(tokens)


def load_transcripts(root: Path, youtuber_filter: Optional[str] = None, debug: bool = False) -> Dict[Tuple[str, str], str]:
    transcripts: Dict[Tuple[str, str], str] = {}
    files_found = 0
    files_loaded = 0
    
    for path in root.rglob("*.txt"):
        if path.is_file():
            files_found += 1
            youtuber = path.parent.name
            video_id = path.stem
            
            if youtuber_filter and youtuber != youtuber_filter:
                continue
            
            text = path.read_text(encoding="utf-8", errors="ignore")
            key = (youtuber, video_id)
            transcripts[key] = text
            files_loaded += 1
            
            if debug:
                print(f"  [DEBUG] Carregada transcrição: {youtuber}/{video_id} ({len(text)} chars)")
    
    if debug:
        print(f"[DEBUG] load_transcripts: {files_found} arquivos encontrados, {files_loaded} carregados")
        if youtuber_filter:
            print(f"[DEBUG] Filtro aplicado: youtuber='{youtuber_filter}'")
    
    return transcripts


def load_comments(root: Path, youtuber_filter: Optional[str] = None, debug: bool = False) -> Dict[Tuple[str, str], List[str]]:
    """
    MODIFICADO: Retorna List[str] (comentários individuais) em vez de str agregada.
    """
    comments: Dict[Tuple[str, str], List[str]] = {}
    if not root.exists():
        if debug:
            print(f"[DEBUG] Diretório de comentários não existe: {root}")
        return comments

    files_found = 0
    files_loaded = 0
    total_comments = 0

    for path in root.rglob("*.txt"):
        if path.is_file():
            files_found += 1
            youtuber = path.parent.name
            video_id = path.stem

            if youtuber_filter and youtuber != youtuber_filter:
                continue

            raw = path.read_text(encoding="utf-8", errors="ignore")
            lines = [line.strip() for line in raw.splitlines() if line.strip()]

            if not lines:
                if debug:
                    print(f"  [DEBUG] Arquivo vazio ignorado: {youtuber}/{video_id}")
                continue

            key = (youtuber, video_id)
            comments[key] = lines  # Lista de comentários individuais
            files_loaded += 1
            total_comments += len(lines)

            if debug:
                print(f"  [DEBUG] Carregados {len(lines)} comentários: {youtuber}/{video_id}")

    if debug:
        print(f"[DEBUG] load_comments: {files_found} arquivos encontrados, {files_loaded} carregados, {total_comments} comentários totais")
        if youtuber_filter:
            print(f"[DEBUG] Filtro aplicado: youtuber='{youtuber_filter}'")

    return comments


def get_all_videos(
    transcripts_root: Path,
    comments_root: Path,
    youtuber_filter: Optional[str] = None,
    debug: bool = False,
) -> List[Tuple[str, str]]:
    """
    Retorna lista de (youtuber, video_id) que têm transcrição ou comentários.
    """
    transcripts = load_transcripts(transcripts_root, youtuber_filter, debug)
    comments = load_comments(comments_root, youtuber_filter, debug)
    
    # União de todos os vídeos
    all_keys = set(transcripts.keys()) & set(comments.keys())
    videos = sorted(list(all_keys))
    
    if debug:
        print(f"[DEBUG] Total de vídeos únicos encontrados: {len(videos)}")
        with_transcript = sum(1 for k in videos if k in transcripts)
        with_comments = sum(1 for k in videos if k in comments)
        with_both = sum(1 for k in videos if k in transcripts and k in comments)
        print(f"[DEBUG]   Com transcrição: {with_transcript}")
        print(f"[DEBUG]   Com comentários: {with_comments}")
        print(f"[DEBUG]   Com ambos: {with_both}")
    
    return videos


def load_video_corpus(
    youtuber: str,
    video_id: str,
    transcripts_root: Path,
    comments_root: Path,
    cleaning_config: CleaningConfig,
    debug: bool = False,
) -> List[VideoDocument]:
    """
    Carrega e limpa os dados de UM vídeo específico.
    """
    docs: List[VideoDocument] = []
    
    # Carregar transcrição
    transcript_path = transcripts_root / youtuber / f"{video_id}.txt"
    if transcript_path.exists():
        raw_text = transcript_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw_text, cleaning_config)
        
        if cleaned and (not cleaning_config.enabled or len(cleaned.split()) >= cleaning_config.min_tokens):
            docs.append(VideoDocument(
                youtuber=youtuber,
                video_id=video_id,
                source="transcript",
                text=cleaned,
                comment_index=None,
            ))
            if debug:
                print(f"  [DEBUG] Transcrição carregada: {len(cleaned.split())} tokens")
        elif debug:
            print(f"  [DEBUG] Transcrição ignorada (vazia ou muito curta)")
    
    # Carregar comentários
    comments_path = comments_root / youtuber / f"{video_id}.txt"
    if comments_path.exists():
        raw = comments_path.read_text(encoding="utf-8", errors="ignore")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        
        for idx, raw_comment in enumerate(lines):
            cleaned = clean_text(raw_comment, cleaning_config)
            
            if not cleaned:
                continue
                
            if cleaning_config.enabled and len(cleaned.split()) < cleaning_config.min_tokens:
                continue
            
            docs.append(VideoDocument(
                youtuber=youtuber,
                video_id=video_id,
                source="comment",
                text=cleaned,
                comment_index=idx,
            ))
        
        if debug:
            comment_docs = [d for d in docs if d.source == "comment"]
            print(f"  [DEBUG] Comentários carregados: {len(comment_docs)} de {len(lines)}")
    
    return docs


def create_topic_model(embedding_model_name: str, debug: bool = False) -> BERTopic:
    if debug:
        print(f"\n[DEBUG] Criando modelo de tópicos...")
        print(f"[DEBUG]   Modelo de embedding: {embedding_model_name}")

    embedding_model = SentenceTransformer(embedding_model_name)

    umap_model = UMAP(
        n_components=15,
        n_neighbors=25,
        min_dist=0.15,
        metric="cosine",
    )

    hdbscan_model = HDBSCAN(
        min_samples=5,
        min_cluster_size=30,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    if debug:
        print(f"[DEBUG]   UMAP: n_components=5, n_neighbors=15")
        print(f"[DEBUG]   HDBSCAN: min_samples=10")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=True,
        language="multilingual",
    )

    if debug:
        print(f"[DEBUG]   Modelo BERTopic criado com sucesso")

    return topic_model


def fit_topics_per_video(
    topic_model: BERTopic,
    docs: List[VideoDocument],
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Treina modelo e retorna tópicos/probabilidades.
    """
    texts = [d.text for d in docs]

    if debug:
        print(f"\n[DEBUG] fit_topics_per_video iniciado")
        print(f"[DEBUG]   Número de documentos: {len(texts)}")
        print(f"[DEBUG]   Tamanho médio do texto: {sum(len(t) for t in texts) / len(texts):.0f} caracteres")

    topics, probs = topic_model.fit_transform(texts)

    if debug:
        unique_topics = set(topics)
        n_topics = len(unique_topics)
        print(f"\n[DEBUG] fit_topics_per_video finalizado")
        print(f"[DEBUG]   Número de tópicos descobertos: {n_topics}")
        print(f"[DEBUG]   IDs dos tópicos: {sorted(unique_topics)}")
        if probs is not None:
            print(f"[DEBUG]   Shape das probabilidades: {probs.shape}")

    return np.asarray(topics), np.asarray(probs) if probs is not None else np.array([])


def compute_video_alignment(
    docs: List[VideoDocument],
    topics: np.ndarray,
    probs: np.ndarray,
    debug: bool = False,
) -> Optional[Dict]:
    """
    Computa métricas de alinhamento para um vídeo.
    Retorna dict com métricas ou None se não for possível calcular.
    """
    # Separar transcrição e comentários
    transcript_indices = [i for i, d in enumerate(docs) if d.source == "transcript"]
    comment_indices = [i for i, d in enumerate(docs) if d.source == "comment"]
    
    if not transcript_indices or not comment_indices:
        if debug:
            print(f"  [DEBUG] Pulando: sem transcrição ou comentários")
        return None
    
    if probs.ndim != 2 or probs.shape[0] == 0:
        if debug:
            print(f"  [DEBUG] Pulando: sem matriz de probabilidades")
        return None
    
    # Calcular distribuição média para transcrição (normalmente só tem 1)
    transcript_dist = probs[transcript_indices].mean(axis=0)
    
    # Calcular distribuição média para comentários
    comment_dist = probs[comment_indices].mean(axis=0)
    
    # Normalizar
    transcript_dist = transcript_dist / max(transcript_dist.sum(), 1e-12)
    comment_dist = comment_dist / max(comment_dist.sum(), 1e-12)
    
    # Calcular métricas
    cos_sim = cosine_similarity(transcript_dist, comment_dist)
    js_div = js_divergence(transcript_dist, comment_dist)
    l1 = l1_distance(transcript_dist, comment_dist)
    
    n_topics = len(transcript_dist)
    
    if debug:
        print(f"  [DEBUG] Métricas computadas:")
        print(f"  [DEBUG]   Tópicos: {n_topics}")
        print(f"  [DEBUG]   Cosine: {cos_sim:.4f}")
        print(f"  [DEBUG]   JS Div: {js_div:.4f}")
        print(f"  [DEBUG]   L1 Dist: {l1:.4f}")
    
    return {
        "cosine_similarity": cos_sim,
        "js_divergence": js_div,
        "l1_distance": l1,
        "n_topics": n_topics,
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.dot(a, b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return num / denom


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(p - q)))


def process_single_video(
    youtuber: str,
    video_id: str,
    transcripts_root: Path,
    comments_root: Path,
    cleaning_config: CleaningConfig,
    embedding_model_name: str,
    debug: bool = False,
) -> Optional[Dict]:
    """
    Processa um único vídeo: carrega dados, treina modelo, computa métricas.
    Retorna dict com resultados ou None em caso de erro.
    """
    if debug:
        print(f"\n{'='*60}")
        print(f"[DEBUG] Processando vídeo: {youtuber}/{video_id}")
        print(f"{'='*60}")
    else:
        print(f"Processando: {youtuber}/{video_id}")
    
    # 1. Carregar corpus do vídeo
    docs = load_video_corpus(
        youtuber, video_id,
        transcripts_root, comments_root,
        cleaning_config, debug
    )
    
    if len(docs) < 2:
        print(f"  ⚠️  Pulando: apenas {len(docs)} documento(s)")
        return None
    
    n_transcripts = sum(1 for d in docs if d.source == "transcript")
    n_comments = sum(1 for d in docs if d.source == "comment")
    
    if n_transcripts == 0 or n_comments == 0:
        print(f"  ⚠️  Pulando: {n_transcripts} transcrições, {n_comments} comentários")
        return None
    
    print(f"  📄 {n_transcripts} transcrição(ões), 💬 {n_comments} comentários")
    
    topic_model = None
    
    try:
        # 2. Criar e treinar modelo para este vídeo
        topic_model = create_topic_model(embedding_model_name, debug=debug)
        topics, probs = fit_topics_per_video(topic_model, docs, debug=debug)
        
        # 3. Computar métricas de alinhamento
        metrics = compute_video_alignment(docs, topics, probs, debug=debug)
        
        if metrics is None:
            return None
        
        result = {
            "youtuber": youtuber,
            "video_id": video_id,
            **metrics,
        }
        
        print(f"  ✅ Concluído: cos_sim={metrics['cosine_similarity']:.3f}, "
              f"n_topics={metrics['n_topics']}")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Erro ao processar vídeo: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return None
    finally:
        # Limpar memória explicitamente
        if topic_model is not None:
            del topic_model
        gc.collect()


def make_cleaning_config(profile: str) -> CleaningConfig:
    if profile == "none":
        return CleaningConfig(enabled=False)

    if profile == "basic":
        return CleaningConfig(
            enabled=True,
            remove_urls=True,
            remove_mentions=True,
            process_hashtags=True,
            remove_numbers=True,
            remove_punctuation=True,
            normalize_elongated=False,
            expand_slang=False,
            remove_stopwords=True,
            min_tokens=1,
        )

    # full
    return CleaningConfig(
        enabled=True,
        remove_urls=True,
        remove_mentions=True,
        process_hashtags=True,
        remove_numbers=True,
        remove_punctuation=True,
        normalize_elongated=True,
        expand_slang=True,
        remove_stopwords=True,
        min_tokens=3,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Modelagem de tópicos por vídeo (BERTopic) "
            "para transcrições e comentários de vídeos do YouTube."
        )
    )

    parser.add_argument(
        "--transcripts-root",
        type=Path,
        default=Path("text"),
        help="Diretório raiz das transcrições: ./text/<youtuber>/<video_id>.txt",
    )

    parser.add_argument(
        "--comments-root",
        type=Path,
        default=Path("comments"),
        help="Diretório raiz dos comentários: ./comments/<youtuber>/<video_id>.txt",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help=(
            "Nome do modelo de sentence embeddings (SentenceTransformers). "
            "Ex.: all-MiniLM-L6-v2, BERTweet, RoBERTa, etc."
        ),
    )

    parser.add_argument(
        "--cleaning-profile",
        type=str,
        choices=["none", "basic", "full"],
        default="full",
        help="Nível de limpeza de texto a aplicar: none | basic | full.",
    )

    parser.add_argument(
        "--output-alignment",
        type=Path,
        default=Path("topic_alignment.csv"),
        help="Caminho para salvar as métricas de alinhamento transcrições vs comentários.",
    )

    parser.add_argument(
        "--youtuber",
        type=str,
        default=None,
        help="Filtrar processamento para um youtuber específico (case-sensitive).",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ativa modo debug com informações detalhadas de processamento.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cleaning_config = make_cleaning_config(args.cleaning_profile)

    if args.debug:
        print("=" * 60)
        print("[DEBUG] MODO DEBUG ATIVADO")
        print("=" * 60)
        print(f"[DEBUG] Argumentos:")
        print(f"[DEBUG]   transcripts_root: {args.transcripts_root}")
        print(f"[DEBUG]   comments_root: {args.comments_root}")
        print(f"[DEBUG]   embedding_model: {args.embedding_model}")
        print(f"[DEBUG]   cleaning_profile: {args.cleaning_profile}")
        print(f"[DEBUG]   youtuber_filter: {args.youtuber}")
        print(f"[DEBUG]   output_alignment: {args.output_alignment}")
        print()

    print("Obtendo lista de vídeos...")
    videos = get_all_videos(
        transcripts_root=args.transcripts_root,
        comments_root=args.comments_root,
        youtuber_filter=args.youtuber,
        debug=args.debug,
    )
    
    if not videos:
        raise RuntimeError("Nenhum vídeo encontrado.")
    
    print(f"Total de vídeos a processar: {len(videos)}")
    print()

    results: List[Dict] = []
    
    for idx, (youtuber, video_id) in enumerate(videos, 1):
        print(f"\n[{idx}/{len(videos)}] ", end="")
        
        result = process_single_video(
            youtuber=youtuber,
            video_id=video_id,
            transcripts_root=args.transcripts_root,
            comments_root=args.comments_root,
            cleaning_config=cleaning_config,
            embedding_model_name=args.embedding_model,
            debug=args.debug,
        )
        
        if result:
            results.append(result)
        
        # Salvar resultados parciais a cada 5 vídeos
        if idx % 5 == 0 and results:
            args.output_alignment.parent.mkdir(parents=True, exist_ok=True)
            df_partial = pd.DataFrame(results)
            df_partial.to_csv(args.output_alignment, index=False)
            print(f"\n  💾 Checkpoint salvo: {len(results)} vídeos processados")
    
    # 3. Salvar resultados finais
    if not results:
        print("\n⚠️  Nenhum vídeo foi processado com sucesso.")
        return
    
    args.output_alignment.parent.mkdir(parents=True, exist_ok=True)
    df_alignment = pd.DataFrame(results)
    df_alignment.to_csv(args.output_alignment, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Processamento concluído!")
    print(f"   Vídeos processados: {len(results)}/{len(videos)}")
    print(f"   Resultados salvos em: {args.output_alignment}")
    print(f"{'='*60}")
    
    # Estatísticas
    print("\nEstatísticas de alinhamento:")
    print(f"  Cosine Similarity: média={df_alignment['cosine_similarity'].mean():.3f}, "
          f"std={df_alignment['cosine_similarity'].std():.3f}")
    print(f"  JS Divergence: média={df_alignment['js_divergence'].mean():.3f}, "
          f"std={df_alignment['js_divergence'].std():.3f}")
    print(f"  L1 Distance: média={df_alignment['l1_distance'].mean():.3f}, "
          f"std={df_alignment['l1_distance'].std():.3f}")
    print(f"  Número médio de tópicos: {df_alignment['n_topics'].mean():.1f}")


if __name__ == "__main__":
    main()
