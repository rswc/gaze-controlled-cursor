# Sterowanie ekranem komputerowym za pomocą wzroku
***
### Zadaniem naszego projektu jest umożliwienie oraz ułatwienie obsługi komputera osobom tymczasowo bądź stale niepełnosprawnym ruchowo jak i cierpiącym na choroby neurologiczne, w tym neurodegeneracyjne. Jest również ciekawe i użyteczne rozwiązanie dla użytkowników zainteresowanych nową technologią i dla tych, którzy zmęczeni są używaniem myszki. Zadaniem programu  jest analiza obrazu przechwyconego przez kamerkę internetową, za pomocą sztucznej inteligencji. Dane jakie jesteśmy w stanie uzyskać z pojedynczych kratek dają możliwość wyznaczania dokładnego punktu na ekranie, na który skierowany jest wzrok użytkownika. 
### Nasz projekt umożliwia aktualnie czytanie książek w wersji elektronicznej, plików, przeglądanie stron internetowych bez potrzeby użycia rąk, dzięki automatycznym sterowaniu ekranem za pomocą wzroku, który kierujemy zależnie, od potrzeby w dół bądź górę strony.
### W przyszłości pragnęlibyśmy stworzyć bardziej precyzyjny model, który byłby w stanie z większą precyzją określać punkt na ekranie na który kierujemy nasz wzrok, co pozwalałby na przeprowadzanie bardziej złożonych i skomplikowanych operacji, jak obsługa całego urządzenia. 
---
### Wymagane moduły:
1. tensorflow w wersji 1.14
2. *numpy
3. *opencv
4. *pytautogui
5. \*openVINO  
-> moduły z '\*' wymagane do uruchomienia skryptu ze sterowaniem wzrokiem (cursor_operator.py)
---
### Instrukcja uruchomienia obsługi kursora
1. Zainstalować wymagane moduły za pomocą pip -m install <nazwa_modułu>
2. Pobrać pakiet openVINO z oficjalnej strony oraz zainstalować go na urządzeniu. W przypadku trudności z dodaniem modułu do PATH wystarczy dodać folder z pakiekietem (openvino/) do Project/scripts/
3. Otworzyć i uruchomić skrypt cursor_operator.py
4. Aby zatrzymać program należy ustawić focus na okienko z kamerą i nacisnąć '[', zostanie wyświetlony komunikat 'PAUSED', aby przywrócić działanie należy drugi raz wcisnąć klawisz '['
5. Zamykanie programu działa analogicznie do pauzowania, z wyjątkiem, że należy dwukrotnie wcisnąć przycisk ']'
Działanie programu polega na automatycznym przewijaniu dokumentu (strony itp.) podczas czytania. Jeżeli wzrok znajduje się w dolnej częsci ekranu tekst przesuwa się w dół i analogicznie w drugą stronę.
---
### Instrukcja uruchomienia algorytmu genetycznego
Działanie implementacji algorytmu genetycznego z genetic_calculator.py zostało zademonstrowane w pliku genetic_test.py.
Po wyliczeniu zadanej liczby pokoleń program poprosi użytkownika o dalsze polecenia. Należy wybrać:
* C - aby kontynuować obliczenia przez podaną liczbę pokoleń
* E - aby edytować zmienne takie jak prawdopodobieństwo mutacji
* Q - aby zakończyć pracę programu
Przebieg obliczeń i wynik będą zapisane w pliku out.txt
---
### Instrukcja uruchomienia kalibracji oraz wskazówki jak wykonać poprawną kalibrację
1. Instalacja modułów jak w obsłudze kursora (wymagane moduły: opencv, numpy, openvino)
2. Otworzyć i uruchomić skrypt calibration.py
3. Zostaną otworzone dwa okienka, jedno pokazujące twarz i nakładane na nią boxy i wektory określonych wartości oraz drugie, w którym odbywa się kalibracja.
4. Na środku ekranu widoczna jest biała kropka na którą należy skierować swój wzrok. Przyciskiem 'x' rozpoczynamy nagrywanie, podczas którego należy skupić wzrok na kropce i poruszać głową w róznych kierunkach (najlepiej dla każdej kolejnej kropki podobnie). Drugie naciśnięcie klawisza 'x' wyłącza nagrywanie i wyświetla nową kropkę.
5. Po skalibrowaniu 9 kropek należy dwukrotnie nacisnąć 'x' aby zakończyć i zapisać wyniki do pliku. Domyślna nazwa pliku to `capresults_J.npy`
6. Należy uruchomić skrypt model_compiler.py, którego wynikiem będą plik .pb oraz `norm.npy`, w folderze `model`
7. Uzyskany plik .pb należy skonwertować za pomocą programu Model Optimizer będącego częścią pakietu OpenVINO. Powstałe pliki należy umieścić w folderze `\models\cursor-estimation-0001\FP32`
---
## Struktura programu: 
* calibration.py - skrypt służący do kalibracji i zbierania danych treningowych. Polega na wyświetlaniu na ekranie punktów, na które użytkownik ma skierować swój wzrok oraz poruszać głową w różnych kierunkach w celu zebrania zróżnicowanych wyników. Dane te zapisywane są w formacie .npy oraz wykorzystywane są później do stworzenia modelu sieci wyznaczającej punkty.

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
---
## Źródła
* [keras-tf-pb](https://github.com/Tony607/keras-tf-pb) by Tony607 przy zapisywaniu modelu w formacie .pb
* [OpenVINO-Python-Utils](https://github.com/simpledevelopments/OpenVINO-Python-Utils) by simpledevelopments, z delikatnymi modyfikacjami, przy ładowaniu modeli OpenVINO
