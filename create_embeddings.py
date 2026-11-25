# Importa gli strumenti per gestire gli argomenti da riga di comando.
import argparse
# Importa Path per lavorare con i percorsi dei file.
import hashlib  # Serve a creare ID stabili a partire dal file e dall'indice del chunk.
from pathlib import Path

# Importa i tipi di configurazione del client Google GenAI.
from google.genai import types

# Importa la funzione di progetto che crea il client Gemini.
from config import get_genai_client
from chroma_config import get_chroma_collection  # Apre la collection persistente di Chroma.
from text_chunking import split_text_into_chunks

# Imposta il modello di embedding scelto per il progetto.
MODEL_NAME = "gemini-embedding-001"
# Definisce il separatore di 30 trattini tra i risultati.
SEPARATOR = "-" * 30


# Definisce il punto di ingresso principale dello script.
def main() -> None:
    # Crea il parser che legge e descrive gli argomenti del comando.
    parser = argparse.ArgumentParser(
        # Spiega brevemente lo scopo dello script nella guida CLI.
        # --- INIZIO MODIFICA: aggiorna la descrizione per indicare il chunking ---
        description="Crea e salva embedding per i chunk dei file di testo indicati."
        # --- FINE MODIFICA: descrizione aggiornata ---
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
    # --- INIZIO PARTE NUOVA: genera e salva un embedding per ogni chunk ---
    for input_path, text in input_texts:  # Elabora ogni file che l'utente ha passato allo script.
        source = str(input_path.resolve())  # Usa il percorso assoluto come riferimento stabile al file.
        chunks = split_text_into_chunks(text)  # Suddivide il contenuto usando i limiti impostati sopra.
        chunk_ids: list[str] = []  # Raccoglie gli ID stabili da assegnare ai chunk.
        embeddings: list[list[float]] = []  # Raccoglie i vettori restituiti da Gemini.
        documents: list[str] = []  # Raccoglie i testi originali da salvare con i vettori.
        metadatas: list[dict[str, str | int]] = []  # Raccoglie la fonte e la posizione di ciascun chunk.

        for chunk_index, (start, end, chunk) in enumerate(chunks):  # Elabora ogni chunk con il suo indice.
            response = client.models.embed_content(  # Richiede a Gemini l'embedding del chunk corrente.
                model=MODEL_NAME,  # Seleziona il modello di embedding configurato per il progetto.
                contents=chunk,  # Invia al modello solo il testo di questo chunk.
                config=types.EmbedContentConfig(  # Imposta il tipo di embedding da generare.
                    task_type="RETRIEVAL_DOCUMENT",  # Ottimizza il vettore per il recupero di documenti.
                    output_dimensionality=768,  # Richiede vettori compatibili con la collection.
                ),  # Conclude la configurazione della richiesta.
            )  # Conclude la chiamata al servizio Gemini.
            if not response.embeddings or not response.embeddings[0].values:  # Verifica che il vettore esista.
                raise RuntimeError(  # Segnala il problema invece di salvare un record incompleto.
                    f"il modello non ha restituito un embedding per "  # Introduce il contesto dell'errore.
                    f"{input_path}, chunk {chunk_index}"  # Identifica il file e il chunk interessati.
                )  # Conclude la costruzione dell'errore.

            chunk_id = hashlib.sha256(  # Crea un ID stabile per questo file e questa posizione.
                f"{source}:{chunk_index}".encode("utf-8")  # Converte il riferimento in byte per l'hash.
            ).hexdigest()  # Ottiene una stringa utilizzabile come ID in Chroma.
            chunk_ids.append(chunk_id)  # Aggiunge l'ID alla lista destinata a Chroma.
            embeddings.append(list(response.embeddings[0].values))  # Conserva il vettore numerico.
            documents.append(chunk)  # Conserva il testo associato al vettore.
            metadatas.append(  # Aggiunge i dati necessari per ritrovare il pezzo nel file.
                {  # Inizia i metadati associati a questo chunk.
                    "source": source,  # Salva il percorso del documento originale.
                    "chunk_index": chunk_index,  # Salva la posizione progressiva del chunk.
                    "char_start": start,  # Salva l'indice del primo carattere nel documento.
                    "char_end": end,  # Salva l'indice esclusivo dopo l'ultimo carattere.
                }  # Conclude i metadati di questo chunk.
            )  # Conclude l'aggiunta dei metadati alla lista.

            print(f"File: {input_path} - chunk {chunk_index + 1}/{len(chunks)}")  # Mostra l'avanzamento.
            print(f"Testo:\n{chunk}")  # Mostra il contenuto effettivamente trasformato.
            print(f"Dimensione embedding: {len(response.embeddings[0].values)}")  # Mostra la dimensione.
            print(SEPARATOR)  # Separa l'output del chunk dal successivo.

        collection.delete(where={"source": source})  # Rimuove eventuali chunk precedenti dello stesso file.
        collection.upsert(  # Inserisce i chunk nuovi o aggiorna quelli con lo stesso ID.
            ids=chunk_ids,  # Associa ogni vettore al suo ID stabile.
            embeddings=embeddings,  # Salva tutti gli embedding generati per il file.
            documents=documents,  # Salva i testi originali dei chunk.
            metadatas=metadatas,  # Salva fonte e posizione di ogni chunk.
        )  # Conclude l'operazione di persistenza in Chroma.
        print(f"Salvati {len(chunks)} chunk nella collection document_chunks")  # Conferma il numero salvato.
    # --- FINE PARTE NUOVA: genera e salva un embedding per ogni chunk ---


# Esegue main solo quando questo file viene avviato direttamente.
if __name__ == "__main__":
    # Avvia la lettura dei file e la generazione degli embedding.
    main()
