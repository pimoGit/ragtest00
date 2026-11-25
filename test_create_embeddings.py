import unittest  # Importa il framework di test incluso nella libreria standard di Python.

from text_chunking import split_text_into_chunks  # Importa la funzione da verificare.


class SplitTextIntoChunksTests(unittest.TestCase):  # Raggruppa i test della funzione di chunking.
    def test_short_text_is_returned_as_one_chunk(self) -> None:  # Verifica il caso di testo più piccolo del limite.
        text = "Testo breve."  # Prepara un documento che non richiede suddivisione.

        self.assertEqual(  # Confronta il risultato con un unico chunk che copre tutto il testo.
            split_text_into_chunks(text),  # Esegue la funzione con i parametri predefiniti.
            [(0, len(text), text)],  # Attende inizio, fine esclusiva e contenuto del chunk.
        )  # Conclude il controllo del risultato.

    def test_long_text_splits_at_sentence_boundaries_with_overlap(self) -> None:  # Verifica più chunk e sovrapposizione.
        text = "Alfa beta gamma. " * 12  # Crea testo abbastanza lungo con confini di frase ripetuti.

        chunks = split_text_into_chunks(text, chunk_size=60, overlap=10)  # Usa valori ridotti per il test.

        self.assertGreater(len(chunks), 1)  # Controlla che il testo sia stato diviso.
        self.assertTrue(chunks[0][2].endswith("."))  # Verifica che il primo chunk termini a fine frase.
        for start, end, chunk in chunks:  # Controlla i contenuti e gli intervalli di ciascun chunk.
            self.assertEqual(text[start:end], chunk)  # Verifica che gli indici puntino al testo corretto.
            self.assertLessEqual(len(chunk), 60)  # Verifica che il chunk non superi il limite impostato.
        self.assertLess(chunks[1][0], chunks[0][1])  # Verifica che due chunk consecutivi si sovrappongano.

    def test_invalid_chunk_parameters_are_rejected(self) -> None:  # Verifica la validazione dei parametri.
        with self.assertRaises(ValueError):  # Si aspetta un errore per una sovrapposizione troppo ampia.
            split_text_into_chunks("testo", chunk_size=10, overlap=10)  # Passa una configurazione non valida.


if __name__ == "__main__":  # Rileva l'esecuzione diretta di questo file.
    unittest.main()  # Avvia i test quando il file viene eseguito come programma.
