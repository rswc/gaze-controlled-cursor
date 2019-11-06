# Obsługa komputera za pomocą wzroku
***
### Stworzyliśmy program wykorzystujący sztuczną inteligencję, którego zdaniem jest analizowanie obrazu z kamerki internetowej, a dane jakie jesteśmy w stanie wyciągnąć z pojedynczych kratek służą do wyznaczania punktu na ekranie, na który skierowany jest wzrok użytkownika.
### Zastosowaniem projektu jest możliwość czytania książek, pdfów, przeglądanie stron bez potrzeby używania rąk, przewijając ekran automatycznie, gdy wzrok zbliża się do dołu/góry strony.
### Celem rozwojowym projektu jest stworzenie bardziej precyzyjnego modelu, który byłby w stanie z większą dokładnością określać punkt na ekranie, aby móc przeprowadzać bardziej skomplikowane operacje za pomocą wzroku.
---
### Wymagane moduły:
1. tensorflow w wersji 1.14
2. *numpy
3. *opencv
4. *pytautogui
5. *openVINO  
-> moduły z '*' wymagane do uruchomienia skryptu ze sterowaniem wzrokiem (cursor_operator.py)
---
### Instrukcja uruchomienia obsługi kursora
1. Zainstalować wymagane moduły za pomocą pip -m install <nazwa_modułu>
2. Pobrać pakiet openVINO z oficjalnej strony oraz zainstalować go na urządzeniu. W przypadku trudności z dodaniem modułu do PATH wystarczy dodać folder z pakiekietem (openvino/) do Project/scripts/
3. Otworzyć i uruchomić skrypt cursor_operator.py
4. Aby zamknąć program należy ustawić focus na okienko z kamerą i nacisnąć spację dwa razy
Działanie programu polega na automatycznym przewijaniu dokumentu (strony itp.) podczas czytania. Jeżeli wzrok znajduje się w dolnej częsci ekranu tekst przesuwa się w dół i analogicznie w drugą stronę.
---
### Instrukcja uruchomienia algorytmu genetycznego
`TODO RAPTOR`
---
### Instrukcja uruchomienia kalibracji oraz wskazówki jak wykonać poprawną kalibrację
1. Instalacja modułów jak w obsłudze kursora (wymagane moduły: opencv, numpy, openvino)
2. Otworzyć i uruchomić skrypt calibration.py
3. Zostaną otworzone dwa okienka, jedno pokazujące twarz i nakładane na nią boxy i wektory określonych wartości oraz drugie, w którym odbywa się kalibracja.
4. Na środku ekranu widoczna jest biała kropka na którą należy skierować swój wzrok. Przyciskiem 'x' rozpoczynamy nagrywanie, podczas którego należy skupić wzrok na kropce i poruszać głową w róznych kierunkach (najlepiej dla każdej kolejnej kropki podobnie). Drugie naciśnięcie klawisza 'x' wyłącza nagrywanie i wyświetla nową kropkę.
5. Po skalibrowaniu 9 kropek należy dwukrotnie nacisnąć 'x' aby zakończyć i zapisać wyniki do pliku.
`TODO RAPTOR`
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
***
## Schemat działania:  
  - Wczytywanie danych z kalibracji
  - Uczenie modeli za pomocą optymalnej sieci
  - Pętla programu
   + Przechwytywanie klatki z obrazu kamerki internetowej
   + Analiza klatki przez odpowiedni skrypt i wyznaczenie wartości wejściowych sieci
   + Wykorzystanie zwróconych punktów do sterowania myszką: 
     - Przesuwanie kursora w segment ekranu odpowiadający punktowi, na który skierowany jest wzrok użytkownika
     - Jeżeli segmentem oglądanym jest część górna/dolna ekranu wywoływane jest polecenie do scrollowania w górę/dół
