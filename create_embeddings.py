# Importa gli strumenti per gestire gli argomenti da riga di comando.
import argparse
# Importa Path per lavorare con i percorsi dei file.
import hashlib
from pathlib import Path

# Importa i tipi di configurazione del client Google GenAI.
from google.genai import types

# Importa la funzione di progetto che crea il client Gemini.
from config import get_genai_client
from chroma_config import get_chroma_collection

# Imposta il modello di embedding scelto per il progetto.
MODEL_NAME = "gemini-embedding-001"
# Definisce il separatore di 30 trattini tra i risultati.
SEPARATOR = "-" * 30


# Definisce il punto di ingresso principale dello script.
def main() -> None:
    # Crea il parser che legge e descrive gli argomenti del comando.
    parser = argparse.ArgumentParser(
        # Spiega brevemente lo scopo dello script nella guida CLI.
        description="Crea e stampa un embedding per ogni file di testo indicato."
    # Conclude la configurazione iniziale del parser.
    )
    # Aggiunge l'argomento obbligatorio per i file da elaborare.
    parser.add_argument(
        # Assegna all'argomento il nome interno "inputs".
        "inputs",
        # Richiede almeno un percorso e accetta più file.
        nargs="+",
        # Converte ogni percorso fornito in un oggetto Path.
        type=Path,
        # Mostra FILE come nome dell'argomento nella guida.
        metavar="FILE",
        # Indica che i file devono contenere testo UTF-8.
        help="percorso di un file di testo UTF-8",
    # Conclude la definizione dell'argomento dei file.
    )
    # Legge dalla riga di comando i percorsi dei file.
    input_paths = parser.parse_args().inputs

    # Prepara una lista per associare ogni percorso al testo letto.
    input_texts: list[tuple[Path, str]] = []
    # Elabora i percorsi specificati dall'utente.
    for input_path in input_paths:
        # Rifiuta percorsi che non corrispondono a file regolari.
        if not input_path.is_file():
            # Mostra l'errore nel formato standard di argparse.
            parser.error(f"il percorso non è un file: {input_path}")
        # Gestisce separatamente la lettura del contenuto.
        try:
            # Legge il file interpretandolo come testo UTF-8.
            text = input_path.read_text(encoding="utf-8")
        # Intercetta errori di lettura o di codifica del testo.
        except (OSError, UnicodeError) as error:
            # Comunica quale file non è stato possibile leggere e perché.
            parser.error(f"impossibile leggere {input_path}: {error}")
        # Controlla che il testo contenga almeno un carattere non vuoto.
        if not text.strip():
            # Interrompe l'esecuzione indicando il file vuoto.
            parser.error(f"il file è vuoto: {input_path}")
        # Conserva il percorso insieme al testo letto.
        input_texts.append((input_path, text))

    # Crea il client Gemini usando la configurazione del progetto.
    client = get_genai_client()
    # Apre la collection persistente senza usare una funzione di embedding di Chroma.
    collection = get_chroma_collection()
    # Genera un embedding per ciascun file letto.
    for input_path, text in input_texts:
        # Invia il testo al servizio di embedding del modello scelto.
        response = client.models.embed_content(
            # Specifica il modello di embedding da utilizzare.
            model=MODEL_NAME,
            # Fornisce al modello il testo letto dal file.
            contents=text,
            # Configura l'embedding per un documento destinato al recupero.
            config=types.EmbedContentConfig(
                # Ottimizza il vettore per la ricerca di documenti.
                task_type="RETRIEVAL_DOCUMENT",
                # Limita il vettore restituito a 768 dimensioni.
                output_dimensionality=768,
            ),
        # Conclude la richiesta di creazione dell'embedding.
        )
        # Verifica che la risposta contenga un vettore non vuoto.
        if not response.embeddings or not response.embeddings[0].values:
            # Segnala esplicitamente che il modello non ha fornito il vettore.
            raise RuntimeError(
                # Include il percorso del file che ha causato il problema.
                f"il modello non ha restituito un embedding per {input_path}"
            # Conclude la costruzione del messaggio di errore.
            )

        # Usa un ID deterministico basato sul percorso così da poter ripetere il salvataggio.
        document_id = hashlib.sha256(
            str(input_path.resolve()).encode("utf-8")
        ).hexdigest()
        embedding = list(response.embeddings[0].values)

        # Memorizza il testo originale, il vettore e il riferimento al file.
        collection.upsert(
            ids=[document_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"source": str(input_path.resolve())}],
        )

        # Stampa il percorso del file associato al risultato.
        print(f"File: {input_path}")
        # Stampa il testo originale passato al modello.
        print(f"Testo:\n{text}")
        # Stampa i valori numerici dell'embedding ricevuto.
        print(f"Embedding:\n{embedding}")
        # Conferma che il documento è stato salvato nella collection persistente.
        print("Stato: salvato nella collection document_chunks")
        # Stampa il separatore che distingue questo risultato dal successivo.
        print(SEPARATOR)


# Esegue main solo quando questo file viene avviato direttamente.
if __name__ == "__main__":
    # Avvia la lettura dei file e la generazione degli embedding.
    main()
