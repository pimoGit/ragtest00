CHUNK_SIZE_CHARS = 700  # Limita ogni chunk a circa 700 caratteri.
CHUNK_OVERLAP_CHARS = 100  # Ripete 100 caratteri tra chunk consecutivi per mantenere contesto.


def split_text_into_chunks(
    text: str,  # Riceve il testo completo da suddividere.
    chunk_size: int = CHUNK_SIZE_CHARS,  # Accetta una dimensione massima personalizzabile.
    overlap: int = CHUNK_OVERLAP_CHARS,  # Accetta una sovrapposizione personalizzabile.
) -> list[tuple[int, int, str]]:  # Restituisce inizio, fine e contenuto di ogni chunk.
    """Divide il testo per caratteri, preferendo paragrafi e frasi."""  # Chiarisce che non conta token.
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:  # Controlla che i parametri siano validi.
        raise ValueError(  # Interrompe l'elaborazione se la configurazione non è valida.
            "chunk_size deve essere positivo e overlap tra 0 e chunk_size - 1"  # Specifica i vincoli richiesti.
        )  # Conclude la costruzione dell'errore.

    chunks: list[tuple[int, int, str]] = []  # Prepara i chunk con i rispettivi intervalli nel testo.
    start = 0  # Inizia la scansione dal primo carattere.

    while start < len(text):  # Continua finché rimane testo da elaborare.
        end = min(start + chunk_size, len(text))  # Imposta il limite senza superare la fine del testo.
        if end < len(text):  # Cerca un punto naturale solo se il chunk non è già l'ultimo.
            minimum_end = start + chunk_size // 2  # Evita di creare chunk troppo corti.
            boundaries = [  # Cerca separatori naturali nella parte finale del chunk.
                (text.rfind("\n\n", minimum_end, end), 2),  # Preferisce una separazione tra paragrafi.
                (text.rfind(". ", minimum_end, end), 2),  # Cerca la fine di una frase.
                (text.rfind("? ", minimum_end, end), 2),  # Riconosce anche le domande.
                (text.rfind("! ", minimum_end, end), 2),  # Riconosce anche le esclamazioni.
            ]  # Conclude l'elenco dei separatori possibili.
            boundary, boundary_length = max(boundaries)  # Sceglie il separatore più vicino al limite.
            if boundary >= minimum_end:  # Usa il separatore se non rende il chunk troppo corto.
                end = boundary + boundary_length  # Include nel chunk il separatore trovato.
            else:  # Se manca un separatore di frase o paragrafo, cerca uno spazio.
                boundary = text.rfind(" ", minimum_end, end)  # Cerca l'ultimo spazio prima del limite.
                if boundary >= minimum_end:  # Usa lo spazio solo se resta un chunk sufficientemente lungo.
                    end = boundary  # Evita di spezzare una parola quando possibile.

        raw_chunk = text[start:end]  # Estrae il segmento corrispondente agli indici calcolati.
        chunk = raw_chunk.strip()  # Rimuove gli spazi vuoti all'inizio e alla fine del segmento.
        if chunk:  # Ignora eventuali segmenti composti soltanto da spazi.
            chunk_start = start + len(raw_chunk) - len(raw_chunk.lstrip())  # Corregge l'inizio dopo la pulizia.
            chunk_end = chunk_start + len(chunk)  # Calcola l'indice finale esclusivo del contenuto.
            chunks.append((chunk_start, chunk_end, chunk))  # Conserva testo e posizione nel documento.

        if end == len(text):  # Controlla se è stato raggiunto l'ultimo carattere.
            break  # Termina senza tentare di creare un altro chunk.

        next_start = max(start + 1, end - overlap)  # Riporta l'inizio indietro per sovrapporre i chunk.
        while next_start < end and not text[next_start].isspace():  # Evita di iniziare nel mezzo di una parola.
            next_start += 1  # Avanza fino al successivo spazio disponibile.
        while next_start < len(text) and text[next_start].isspace():  # Salta spazi e ritorni a capo.
            next_start += 1  # Posiziona il nuovo chunk sul primo carattere utile.
        start = next_start  # Riprende la scansione dal punto scelto.

    return chunks  # Restituisce i chunk pronti per essere trasformati in embedding.
