---
name: rag-fundamentals
description: Guida gli agenti nello sviluppo didattico e incrementale di un'app RAG Python che indicizza documenti testuali da una cartella in un database vettoriale e supporta la ricerca semantica, senza frontend.
---

# Fondamenti RAG per questo progetto

## Scopo

Aiutare a realizzare, un passo alla volta, un'applicazione didattica per capire il flusso essenziale di Retrieval-Augmented Generation (RAG). La prima fase termina con la ricerca semantica: non richiede ancora generazione di risposte tramite un LLM.

## Flusso di riferimento

1. Individuare e leggere file testuali da una cartella.
2. Dividere i documenti in unità di testo ricercabili (chunk) prima di indicizzarli nel database vettoriale.
3. Trasformare i testi in embedding.
4. Salvare embedding e metadati associati in un database vettoriale.
5. Trasformare una query in embedding e recuperare i contenuti semanticamente più vicini.

Conservare metadati sufficienti per risalire al documento originale. Evitare di confondere la similarità semantica con una ricerca per corrispondenza letterale.

## Strategia di chunking

Quando si implementa l'indicizzazione nel database vettoriale, ogni documento deve essere diviso in chunk ricercabili invece di creare un unico embedding per l'intero file.

- Un chunk non è un singolo token: è una sequenza coerente di testo, in genere compresa tra circa 500 e 800 token.
- Per chiarezza didattica usare per primi 500-800 caratteri con bordi preferibilmente cercati su paragrafi o frasi, senza introdurre una libreria di parsing nel primo incremento.
- L'implementazione definitiva può contare i token con il tokenizer del provider o con una strategia compatibile; una separazione basata esclusivamente sui caratteri non va presentata come conteggio reale dei token.
- Usare una sovrapposizione di circa 50-100 token per evitare di perdere il contesto vicino al bordo di un chunk.
- Normalmente le dimensioni proposte sono un punto di partenza: devono essere validate in base alla lunghezza dei documenti e alle query reali.
- Salvare per ogni chunk almeno il percorso del documento, il testo e l'indice/posizione del chunk; conservare anche gli indici dei caratteri quando permettono di individuare con precisione il frammento originale.
- Non aggiungere chunking finché il passo corrente richiede soltanto la generazione e la stampa degli embedding: iniziare a crearli quando si aggiunge il database vettoriale.

## Modello di embedding scelto

Per questo progetto usare il modello esterno `gemini-embedding-001` tramite Gemini API. Per il recupero, codificare i documenti con `RETRIEVAL_DOCUMENT` e le query con `RETRIEVAL_QUERY`. Installare le dipendenze da `requirements.txt`, copiare `.env.example` in `.env` e ottenere il client tramite `config.get_genai_client()`. Le chiamate API richiedono una chiave: non inserirla nel codice o nel repository e non versionare `.env`. Il tier gratuito può usare i contenuti inviati per migliorare i prodotti Google: durante gli esperimenti usare solo documenti non sensibili.

## Database vettoriale

Usare ChromaDB in modalità persistente locale. La configurazione è in `chroma_config.py`: `get_chroma_collection()` restituisce la collection `document_chunks` salvata nella cartella locale `chroma_db/`, esclusa dal controllo versione. Gli embedding sono generati da Gemini, quindi la collection non deve usare la funzione di embedding predefinita di Chroma.

## Approccio didattico e vincoli

- Implementare soltanto il requisito del passo richiesto; non costruire in anticipo il flusso RAG completo.
- Preferire funzioni e script Python semplici, con responsabilità chiare e facili da seguire.
- Per ogni passo, chiarire brevemente input, output e ruolo nel flusso.
- Non aggiungere frontend, API, framework o infrastruttura non richiesti.
- Non assumere scelte per database vettoriale, formati dei file, strategia di chunking o gestione delle credenziali senza una richiesta o una decisione esplicita.
- Gestire gli errori in modo esplicito, soprattutto per cartelle inesistenti, file non leggibili e dipendenze/configurazioni mancanti.
- Aggiungere dipendenze e test solo quando necessari per il comportamento richiesto e secondo le convenzioni già presenti nel progetto.
