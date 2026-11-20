---
name: rag-fundamentals
description: Guida gli agenti nello sviluppo didattico e incrementale di un'app RAG Python che indicizza documenti testuali da una cartella in un database vettoriale e supporta la ricerca semantica, senza frontend.
---

# Fondamenti RAG per questo progetto

## Scopo

Aiutare a realizzare, un passo alla volta, un'applicazione didattica per capire il flusso essenziale di Retrieval-Augmented Generation (RAG). La prima fase termina con la ricerca semantica: non richiede ancora generazione di risposte tramite un LLM.

## Flusso di riferimento

1. Individuare e leggere file testuali da una cartella.
2. Dividere i documenti in unità di testo ricercabili (chunk), quando richiesto dal passo corrente.
3. Trasformare i testi in embedding.
4. Salvare embedding e metadati associati in un database vettoriale.
5. Trasformare una query in embedding e recuperare i contenuti semanticamente più vicini.

Conservare metadati sufficienti per risalire al documento originale. Evitare di confondere la similarità semantica con una ricerca per corrispondenza letterale.

## Modello di embedding scelto

Per questo progetto usare il modello esterno `gemini-embedding-001` tramite Gemini API. Per il recupero, codificare i documenti con `RETRIEVAL_DOCUMENT` e le query con `RETRIEVAL_QUERY`. Installare le dipendenze da `requirements.txt`, copiare `.env.example` in `.env` e ottenere il client tramite `config.get_genai_client()`. Le chiamate API richiedono una chiave: non inserirla nel codice o nel repository e non versionare `.env`. Il tier gratuito può usare i contenuti inviati per migliorare i prodotti Google: durante gli esperimenti usare solo documenti non sensibili.

## Approccio didattico e vincoli

- Implementare soltanto il requisito del passo richiesto; non costruire in anticipo il flusso RAG completo.
- Preferire funzioni e script Python semplici, con responsabilità chiare e facili da seguire.
- Per ogni passo, chiarire brevemente input, output e ruolo nel flusso.
- Non aggiungere frontend, API, framework o infrastruttura non richiesti.
- Non assumere scelte per database vettoriale, formati dei file, strategia di chunking o gestione delle credenziali senza una richiesta o una decisione esplicita.
- Gestire gli errori in modo esplicito, soprattutto per cartelle inesistenti, file non leggibili e dipendenze/configurazioni mancanti.
- Aggiungere dipendenze e test solo quando necessari per il comportamento richiesto e secondo le convenzioni già presenti nel progetto.
