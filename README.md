# Obsługa komputera za pomocą wzroku
***
### Stworzyliśmy program wykorzystujący sztuczną inteligencję, którego zdaniem jest analizowanie obrazu z kamerki internetowej, a dane jakie jesteśmy w stanie wyciągnąć z pojedynczych kratek służą do wyznaczania punktu na ekranie, na który skierowany jest wzrok użytkownika.
### Zastosowaniem projektu jest możliwość czytania książek, pdfów, przeglądanie stron bez potrzeby używania rąk, przewijając ekran automatycznie, gdy wzrok zbliża się do dołu/góry strony.
### Celem rozwojowym projektu jest stworzenie bardziej precyzyjnego modelu, który byłby w stanie z większą dokładnością określać punkt na ekranie, aby móc przeprowadzać bardziej skomplikowane operacje za pomocą wzroku.
---
### Wymagane moduły:
1. *tensorflow
2. *numpy
3. *cv2
4. *pytautogui
5. openVINO  
//moduły z '*' wymagane do uruchomienia skryptu ze sterowaniem wzrokiem
---
## Struktura programu: 
*  calibration.py - jest to skrypt służący do kalibracji i zbierania danych treningowych. Polega na wyświetlaniu na ekranie punktów, na które użytkownik ma skierować swój wzrok oraz poruszać głową w różnych kierunkach w celu zebrania zróżnicowanych wyników. Dane te zapisywane są w formacie .npy oraz wykorzystywane są później do stworzenia modelu sieci wyznaczającej punkty.

* test.py - skrypt służący do testowania pojedynczych konfiguracji sieci, aby sprawdzać poprawność modelu.

* model_calculator.py - kalkulator modeli, służy do wyznaczania średniego absolutnego błędu w zależności od konfiguracji warstw, neuronów, epochów itp. Działa na zasadzie dokładnego wyliczania każdej możliwej konfiguracji z wybranych wartości. Na koniec tworzy plik .npy zawierający tabelę konfiguracji wszystkich sieci przeznaczoną do analizy i wyznaczania statystyk.

* cap_combiner - krótki skrypt łączący pliki z danymi z kalibracji. 

* genetic_calculator.py - implementacja genetycznego algorytmu mająca na celu znajdowanie najbardziej optymalnego modelu sieci dla zebranych danych.

* face_processing.py - moduł pomocniczy służący do wykonywania wszystkich skomplikowanych obliczeń potrzebnych do analizy obrazu za pomocą bibliotek openVINO i przekazywania tych danych do treningu modelu.

* analysis_jup.ipynb - plik stworzony przy pomocy platformy jupyter służący do analizy i wizualizacji danych.

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
