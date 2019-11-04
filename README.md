# Obsługa komputera za pomocą wzroku
***
### Stworzyliśmy program wykorzystujący sztuczną inteligencję, którego zdaniem jest analizowanie obrazu z kamerki internetowej.

### Dane jakie jesteśmy w stanie wyciągnąć z pojedynczych kratek służą do wyznaczania punktu na ekranie, na który skierowany jest wzrok użytkownika.

### Zastosowaniem projektu jest możliwość czytania książek, pdfów, przeglądanie stron bez potrzeby używania rąk.
---
## Struktura programu: 
*  calibration.py - jest to skrypt służący do kalibracji i zbierania danych treningowych. Polega na wyświetlaniu na ekranie punktów, na które użytkownik ma skierować swój wzrok oraz poruszać głową w różnych kierunkach w celu zebrania zróżnicowanych wyników. Dane te zapisywane są w formacie .npy oraz wykorzystywane są później do stworzenia modelu sieci wyznaczającej punkty.

* test.py - skrypt służący do testowania pojedynczych konfiguracji sieci, aby sprawdzać poprawność modelu.

* model_calculator.py - kalkulator modeli, służy do wyznaczania średniego absolutnego błędu w zależności od konfiguracji warstw, neuronów, epochów itp. Działa na zasadzie dokładnego wyliczania każdej możliwej konfiguracji z wybranych wartości. Na koniec tworzy plik .npy zawierający tabelę konfiguracji wszystkich sieci przeznaczoną do analizy i wyznaczania statystyk.

* cap_combiner - krótki skrypt łączący pliki z danymi z kalibracji. 

* genetic_calculator.py - ```  - TODO ```

* face_processing.py - ```  - TODO ``` 

* JUPYTER STATYSTKA  ``` + TODO ```

* cursor_operator.py  - ostatni finalny skrypt pythonowy reprezentujący działanie naszego programu.  
Schemat działania:  
  - Wczytywanie danych z kalibracji
  - Uczenie modeli za pomocą optymalnej sieci
  - Pętla programu
   + Przechwytywanie klatki z obrazu kamerki internetowej
   + Analiza klatki przez odpowiedni skrypt i wyznaczenie wartości wejściowych sieci
   + Wykorzystanie zwróconych punktów do sterowania myszką: 
     - Przesuwanie kursora w segment ekranu odpowiadający punktowi, na który skierowany jest wzrok użytkownika
     - Jeżeli segmentem oglądanym jest część górna/dolna ekranu wywoływane jest polecenie do scrollowania w górę/dół
