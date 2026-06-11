# Porównanie implementacji CNN (FashionMNIST)

Dokument zbiera różnice między bliźniaczymi notatnikami implementującymi tę samą
sieć konwolucyjną na zbiorze **FashionMNIST** oraz ich wyniki. Wszystkie notatniki
realizują *identyczny* model i te same hiperparametry — różnią się jedynie
biblioteką/frameworkiem oraz wynikającymi z niej szczegółami API.

Notatniki:

- `AWID-2026-CNN-AWIDNN.ipynb` – własna biblioteka `AWIDNN` (Julia),
- `AWID-2026-CNN-FLUX.ipynb` – `Flux` (Julia) — wersja referencyjna,
- `AWID-2026-CNN-PYTORCH.ipynb` – `PyTorch` (Python),
- `AWID-2026-CNN-TENSORFLOW.ipynb` – `TensorFlow` (niskopoziomowo, Python),
- `AWID-2026-CNN-KERAS.ipynb` – `Keras` (wysokopoziomowo, Python).

## 1. Wspólna specyfikacja

### Architektura sieci (67 708 parametrów)


| Warstwa      | Konfiguracja                               | Parametry |
| ------------ | ------------------------------------------ | --------- |
| Conv 3×3     | 1 → 6 kanałów, padding=1/"same", bez bias  | 54        |
| MaxPool 2×2  | 28×28 → 14×14                              | 0         |
| Conv 3×3     | 6 → 16 kanałów, padding=1/"same", bez bias | 864       |
| MaxPool 2×2  | 14×14 → 7×7                                | 0         |
| Flatten      | 16×7×7 → 784                               | 0         |
| Dense + ReLU | 784 → 84                                   | 65 940    |
| Dropout      | p = 0.4                                    | 0         |
| Dense        | 84 → 10 (logity)                           | 850       |


### Hiperparametry i dane

- Optymalizator: zwykły **SGD**, `lr = 1e-2` (Flux: `Descent`).
- Funkcja straty: **cross-entropy na logitach** (warstwa wyjściowa bez softmax).
- `batchsize = 10`, `epochs = 3`, tasowanie danych w każdej epoce.
- Dane: FashionMNIST, obrazy 28×28 w skali szarości, znormalizowane do `[0, 1]`.

## 2. Niezbędne różnice w implementacjach

### Źródło i format danych


|            | Źródło danych                                    | Typ/zakres                        | Układ tensora    |
| ---------- | ------------------------------------------------ | --------------------------------- | ---------------- |
| Flux       | `MLDatasets.FashionMNIST`                        | `Float32`, `[0,1]`                | WHCN (28×28×1×N) |
| AWIDNN     | `MLDatasets.FashionMNIST`                        | `Float32`, `[0,1]`                | WHCN (28×28×1×N) |
| PyTorch    | `torchvision.datasets.FashionMNIST` + `ToTensor` | `float32`, `[0,1]`                | NCHW (N×1×28×28) |
| TensorFlow | `keras.datasets.fashion_mnist`                   | `uint8`, `0..255` → ręczne `/255` | NHWC (N×28×28×1) |
| Keras      | `keras.datasets.fashion_mnist`                   | `uint8`, `0..255` → ręczne `/255` | NHWC (N×28×28×1) |


Różny układ wymiarów (WHCN/NCHW/NHWC) wpływa na kolejność spłaszczania (`Flatten`),
ale nie zmienia liczby parametrów ani działania sieci.

### Definicja warstw

- **Konwolucja**: `Conv((3,3),1=>6,pad=1,bias=false)` (Flux) ≡
`nn.Conv2d(1,6,3,padding=1,bias=False)` (PyTorch) ≡
`Conv2D(6,3,padding="same",use_bias=False)` (TF). `padding=1` dla jądra 3×3 jest
równoważne `"same"`.
- **Dense z aktywacją**: we Flux i Keras/TF aktywacja `relu` jest częścią warstwy
`Dense`; w PyTorch `nn.ReLU()` jest osobnym elementem `Sequential`.

### Strata, optymalizator, pętla treningowa


|            | Strata (logity)                                   | Optymalizator    | Krok treningu                                          |
| ---------- | ------------------------------------------------- | ---------------- | ------------------------------------------------------ |
| Flux       | `Flux.logitcrossentropy`                          | `Descent(eta)`   | `Flux.gradient` + `Flux.update!`                       |
| PyTorch    | `functional.cross_entropy`                        | `optim.SGD`      | `loss.backward()` + `step()` + `zero_grad()`           |
| TensorFlow | `SparseCategoricalCrossentropy(from_logits=True)` | `optimizers.SGD` | `tf.GradientTape` + `apply_gradients` (`@tf.function`) |
| Keras      | `SparseCategoricalCrossentropy(from_logits=True)` | `optimizers.SGD` | `model.compile` + `model.fit` (pętla ukryta)           |


**Najważniejsza różnica TensorFlow vs Keras**: notatnik TensorFlow implementuje
*własną* pętlę treningową (jawny `GradientTape`, ręczne stosowanie gradientów),
natomiast notatnik Keras używa *wysokopoziomowego* API — `model.compile` (konfiguracja
optymalizatora, straty i metryk) oraz `model.fit`/`model.evaluate` (pętla treningowa
i ewaluacja są ukryte wewnątrz biblioteki). Oba korzystają z tego samego backendu
TensorFlow i dają zgodne wyniki.

- **Etykiety**: Flux używa kodowania one-hot (`onehotbatch`); PyTorch i TF używają
etykiet całkowitych (`sparse`/`*crossentropy`).
- **`onecold`**: we Flux `Flux.onecold`; w Pythonie odpowiednikiem jest `argmax`.

### Wyświetlanie modelu i wykresów

- **Podgląd modelu**: Flux drukuje `Chain` z licznikiem parametrów per warstwa i
sumą; TF `model.summary()` drukuje tabelę z parametrami i sumą; PyTorch wypisuje
listę modułów (bez liczby parametrów).
- **Wykresy**: Flux używa `CairoMakie`, notatniki Pythona używają `matplotlib`
(te same etykiety osi i zakresy, aby wynik wyglądał tak samo).

### Powtarzalność

Różnice w **dokładności przed treningiem** oraz drobne różnice w przebiegu uczenia
wynikają z odmiennych domyślnych schematów inicjalizacji wag w każdym frameworku
oraz z losowego tasowania danych. Architektura, liczba parametrów i końcowa
jakość modelu pozostają zgodne.

## 3. Wyniki

Dokładność (%) na zbiorze treningowym / testowym po każdej epoce. „Przed treningiem"
to dokładność na zbiorze testowym przed rozpoczęciem uczenia (oczekiwane ~10%).


| Implementacja     | Przed treningiem | Epoka 1 (train/test) | Epoka 2 (train/test) | Epoka 3 (train/test) | Czas/epokę |
| ----------------- | ---------------- | -------------------- | -------------------- | -------------------- | ---------- |
| Flux (referencja) | 17.80            | 84.18 / 83.06        | 86.87 / 85.83        | 87.22 / 86.01        | ~11 s      |
| PyTorch           | 8.97             | 81.62 / 80.76        | 84.15 / 82.64        | 87.29 / 86.21        | ~20 s      |
| TensorFlow        | 13.97            | 83.63 / 82.83        | 86.56 / 85.57        | 87.39 / 86.17        | ~13 s      |
| Keras             | 16.51            | 84.21 / 83.18        | 86.74 / 86.20        | 87.91 / 86.90        | ~14 s      |
| AWIDNN            | 12.75            | 84.20 / 83.36        | 86.71 / 85.78        | 88.02 / 86.76        | ~150–190 s |


> Czasy zmierzono na CPU; mogą się różnić w zależności od maszyny.

Wszystkie pięć implementacji osiąga **~86% dokładności na zbiorze
testowym** po 3 epokach, co potwierdza zgodność modeli.

## 4. AWIDNN vs Flux — różnice i ich przyczyny

`AWIDNN` to **własna, dydaktyczna biblioteka w Julii**, która świadomie odwzorowuje
*API* Fluxa (te same nazwy: `Chain`, `Conv`, `MaxPool`, `flatten`, `Dense`,
`Dropout`, `DataLoader`, `logitcrossentropy`, `onehotbatch`/`onecold`,
`Descent`/`setup`/`update!`). Dzięki temu kod obu notatników wygląda niemal
identycznie. Różnice nie leżą więc w interfejsie, lecz w **sposobie działania
silnika** (różniczkowanie, wykonanie, implementacja operacji).

### 4.1. Architektura silnika (najważniejsza różnica)


| Aspekt               | Flux                                                                              | AWIDNN                                                                                                                                | Z czego to wynika                                                                                                                              |
| -------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Różniczkowanie       | `Zygote` — automatyczne AD typu *source-to-source*, różniczkuje dowolny kod Julii | Statyczny **graf obliczeniowy** (DAG węzłów `GraphNode`) z **ręcznie napisanymi** `forward`/`backward` dla każdego operatora          | AWIDNN to implementacja „od zera" w celach dydaktycznych — gradienty wyprowadzono ręcznie z reguły łańcuchowej; Flux generuje je automatycznie |
| Reprezentacja modelu | `net(x)` liczy wynik bezpośrednio (tablica → tablica)                             | `net(input)` buduje **graf** (`Constant`/`Variable` → `BroadcastedOperator`); wartości dostępne dopiero po `forward!` przez `.output` | W AWIDNN wywołanie modelu *konstruuje* graf zamiast od razu liczyć — to on jest „programem", który wykonują `forward!`/`backward!`             |
| Parametry            | ukryte w warstwach, zbierane przez Zygote                                         | jawne liście `Variable` współdzielące pamięć z tablicami warstw                                                                       | `optimize!` aktualizuje te same tablice in-place (`.output .-= η·grad`)                                                                        |


### 4.2. Pętla treningowa i optymalizator


| Krok                 | Flux                                                          | AWIDNN                                                           | Z czego to wynika                                                                                                             |
| -------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Obliczenie gradientu | `grads = Flux.gradient(m -> logitcrossentropy(m(x), y), net)` | `forward!(g)` → `backward!(g)` na zbudowanym wcześniej grafie    | Graf jest budowany **raz** przed pętlą; w iteracji tylko podmieniane są dane (`input.output .= x`) — bez przebudowy topologii |
| Aktualizacja wag     | `Flux.update!(opt_state, net, grads[1])`                      | `AWIDNN.update!(opt_state)` → `optimize!(graph, η)`              | `setup` w AWIDNN owija **graf**, nie model; `update!` ignoruje argumenty model/grads i robi krok SGD po `Variable` w grafie   |
| Stan optymalizatora  | `setup(Descent(η), net)` (na modelu)                          | `setup(Descent(η), g_train)` (na grafie)                         | Krok SGD iteruje po węzłach grafu, a nie po strukturze modelu                                                                 |
| Dropout              | RNG przechwytywany przez Zygote w trakcie różniczkowania      | maska cache'owana w polu warstwy między `forward!` a `backward!` | Statyczny `backward!` musi użyć **dokładnie tej samej** maski co `forward!`, więc jest ona zapamiętywana ręcznie              |


### 4.3. Implementacja operacji (konwolucja / pooling)


| Operacja                  | Flux                                                                | AWIDNN                                                                    | Z czego to wynika                                                                                                                           |
| ------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `Conv` (forward/backward) | zoptymalizowane rdzenie `NNlib` (im2col + BLAS, wielowątkowość/GPU) | naiwny splot na **7 zagnieżdżonych pętlach** (4 zewnętrzne: `b,cout,h,w` × 3 wewnętrzne: `cin,kh,kw`) w czystej Julii | nacisk na czytelność wyprowadzenia gradientu kosztem wydajności (w kodzie wprost zaznaczono, że „docelowo powinno być im2col+gemm lub FFT") |
| Konwencja splotu          | splot matematyczny (flip jądra); `CrossCor` to osobna warstwa       | flip jądra: indeksowanie `W[kH+1-kh, kW+1-kw, …]` — zgodne z `Flux.Conv`  | celowa zgodność z Flux; PyTorch/TF/Keras stosują korelację krzyżową (bez flipu) — dla uczonych wag bez znaczenia funkcjonalnego             |
| `MaxPool`                 | rdzenie `NNlib`                                                     | ręczne pętle; argmax liczony ponownie w `backward` zamiast zapamiętywany  | oszczędność pamięci kosztem niewielkiego narzutu CPU                                                                                        |
| `Dense` (`W*x`)           | BLAS przez `*`                                                      | również BLAS przez `*` (`backward`: `g*Bᵀ`, `Aᵀ*g`)                       | mnożenie macierzowe Julii deleguje do BLAS — warstwa Dense nie jest wąskim gardłem                                                          |
| `logitcrossentropy`       | implementacja `Flux`/`NNlib`                                        | własna, numerycznie stabilna (przesunięcie o `max`), uśredniona po batchu | odpowiednik krok-w-krok, z jawnie wyprowadzonym gradientem `(softmax − y)/B`                                                                |


### 4.4. Inicjalizacja, dane i wyświetlanie


| Aspekt                                       | Flux                                   | AWIDNN                                                                    | Z czego to wynika                                                                                                            |
| -------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Inicjalizacja wag                            | Glorot/Xavier (uniform)                | `Dense`: `randn/√fan_in`; `Conv`: `randn/√(kH·kW·Cin)` (rozkład normalny); w `layers.jl` komentarz TODO wskazuje docelowo Xavier | inny domyślny schemat → inna dokładność początkowa (12.75% vs 17.80%) i nieco inny przebieg uczenia, lecz zbliżona zbieżność |
| Dane / etykiety                              | `MLDatasets`, układ WHCN, one-hot      | identycznie (`MLDatasets`, WHCN, `onehotbatch`)                           | AWIDNN celowo zgodne z Flux — w przeciwieństwie do notatników Pythona nie ma różnic w układzie danych                        |
| Podgląd modelu                               | `Chain` z licznikiem parametrów i sumą | domyślna reprezentacja `Chain` (bez tabeli parametrów)                    | brak dedykowanego `show` z liczbą parametrów (cecha biblioteki dydaktycznej)                                                 |
| Pozostałe wyniki (`@show`, `@info`, wykresy) | —                                      | takie same jak we Flux (`CairoMakie`, te same komunikaty)                 | notatnik AWIDNN jest bezpośrednim odpowiednikiem notatnika Flux                                                              |


### 4.5. Wydajność

Trening AWIDNN trwa **~150–190 s na epokę** wobec ~ 11 s we Flux (kilkunastokrotnie
wolniej). Przyczyną jest naiwna, pętlowa konwolucja w czystej Julii zamiast
zoptymalizowanych rdzeni `NNlib` (BLAS/im2col). Mimo to model osiąga
**~86.8% dokładności testowej** po 3 epokach, zgodnie z pozostałymi implementacjami —
co potwierdza poprawność ręcznie wyprowadzonego różniczkowania.

## 5. Keras

Notatnik `AWID-2026-CNN-KERAS.ipynb` korzysta z wysokopoziomowego API Keras 3
(backend TensorFlow), co jest jego główną cechą odróżniającą od notatnika TensorFlow.

- **Różnice względem niskopoziomowego TensorFlow**: zamiast jawnej pętli
z `tf.GradientTape` model jest trenowany przez `model.fit`, a oceniany przez
`model.evaluate`. Konfiguracja optymalizatora, funkcji straty i metryk odbywa się
jednym wywołaniem `model.compile`. Pętla treningowa, tasowanie i podział na batche
są ukryte wewnątrz biblioteki.
- **Definicja modelu i kompilacja**: identyczna architektura `keras.Sequential`
jak w TensorFlow; `compile(optimizer=SGD(1e-2), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])`.
- **Sposób wyświetlania wyników**: `net.summary()` (tabela z parametrami i sumą),
a wyniki per-epoka i wykresy formatowane tak samo jak w pozostałych notatnikach
Pythona (te same komunikaty `┌ Info ┐`, wykresy `matplotlib`). Trening uruchamiany
epoka po epoce (`model.fit(..., epochs=1, verbose=0)`), aby zachować identyczny
format wypisywania czasu i dokładności.
- **Środowisko**: ten sam venv i kernel `Python (AWID)` co notatniki PyTorch
i TensorFlow (Keras 3 instalowany razem z `tensorflow`).
- **Wyniki**: ~86.9% dokładności na zbiorze testowym po 3 epokach (zob. tabela w sekcji 3).

## 6. Kontekst i rekomendacje optymalizacji AWIDNN

> Ta sekcja zbiera pełny kontekst potrzebny do optymalizacji wydajności biblioteki
> `AWIDNN` (`packages/AWIDNN/src/`). Cel: skrócić czas epoki z ~ 150–190 s w stronę
> rzędu wielkości Fluxa (~ 11 s), **bez zmiany wyników** (~ 86.8% test acc) i **bez
> łamania zgodności API** z notatnikiem `AWID-2026-CNN-AWIDNN.ipynb`.

### 6.1. Mapa kodu (gdzie co się dzieje)


| Plik            | Zawartość                                                                                      | Znaczenie dla wydajności                               |
| --------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `structures.jl` | typy grafu (`GraphNode`, `Variable`, `BroadcastedOperator`), warstwy, `DataLoader`             | **typy pól** decydują o stabilności typów i alokacjach |
| `autodiff.jl`   | `graph`/`forward!`/`backward!`/`optimize!`, `forward`/`backward` per operator, splot i pooling | **gorący kod** — tu powstaje ~99% czasu i alokacji     |
| `layers.jl`     | konstruktory warstw, budowa grafu, inicjalizacja wag, `DataLoader` (konstruktor + iteracja)    | inicjalizacja, mapowanie warstwa → graf, kopie batchy  |

Martwy kod dla tego benchmarku (nie optymalizować): `Embedding` (sama struktura,
bez mapowania na graf), `ScalarOperator` (nieużywany w ścieżce CNN), `bce`/`bce_el`/
`sum_node` (strata CNN to `logitcrossentropy`), `σ` (sieć używa tylko `relu`).


### 6.2. Profil obciążenia (pomiary z notatnika)

- ~**150–190 s/epokę** (Flux ~11 s) — różnica ~15×.
- ~**10 GiB alokacji na epokę**, ~3,3 mln alokacji (`@time` w komórce 3) → silna presja na garbage collector.
- Dominują dwie funkcje w `autodiff.jl`: `_conv_forward` i `_conv_backward`
  (~7-krotnie zagnieżdżone pętle wykonywane 2× na batch × 6000 batchy).
- Alokacje to suma świeżych tablic per operator per batch (wyjścia forward,
  gradienty backward, zera z `zerograd!`, maska Dropoutu, kopie batchy
  z `DataLoader`) — rzędu ~1,5 MB/batch × 6000 batchy ≈ 9–10 GiB.

### 6.3. Zidentyfikowane wąskie gardła (z odwołaniem do kodu)


| #   | Wąskie gardło                                | Lokalizacja                                                                                                | Przyczyna                                                                                    |
| --- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| B1  | Naiwny splot 7-krotnie zagnieżdżonych pętli  | `_conv_forward`, `_conv_backward`                                                                          | złożoność `O(B·Cout·Hout·Wout·Cin·kH·kW)` na indeksowaniu skalarnym, bez BLAS/SIMD; najszybciej zmienna pętla wewnętrzna (`kw`→`wi`) chodzi po **drugim** wymiarze tablicy — dostęp niezgodny z układem column-major |
| B2  | ~~Typy abstrakcyjne w węzłach grafu~~ **✅ rozwiązane (B2)** | `BroadcastedOperator` z inferencją `T,G` (`_infer_op_types`, `_storage_array_type`) | ~~`NodeValue` na wszystkich operatorach~~ → konkretne `Array`/`Matrix`/`Float32`; zapasowy `NodeValue` gdy inferencja niemożliwa; czas epoki bez istotnej zmiany (dispatch ~20 węzłów był pomijalny) |
| B3  | Alokacja wyniku w każdym `forward`           | `_compute!`: `n.output = forward(...)`                                                                     | nowe tablice w każdej iteracji zamiast zapisu do bufora in-place                             |
| B4  | Akumulacja gradientu nie-in-place            | `_accumulate!`: `n.gradient = n.gradient .+ g`                                                             | każda akumulacja alokuje nową tablicę (docelowo `.+=`, ale zob. ostrzeżenie o aliasingu w 6.5) |
| B5  | `zerograd!` alokuje zera co krok             | `_zerograd!(::Variable)` = `zero(output)`                                                                  | bufory gradientów parametrów tworzone od nowa w każdym `backward!` (≈0,3 MB/batch)           |
| B6  | ~~Ponowne liczenie argmax w `backward` MaxPool~~ **✅ rozwiązane (B6)** | `_maxpool_backward_into!` (cache `:ihi`/`:iwi` z forward)                                                   | ~~drugie przejście okna~~ → scatter z cache; wariant `_maxpool_backward_recompute_into!` zachowany do testów |
| B7  | Brak wielowątkowości w splocie i poolingu    | `_conv_*`, `_maxpool_*`                                                                                    | pętle jednowątkowe bez `Threads.@threads`/`@simd`; jedynie `Dense` (`*`) idzie przez BLAS    |
| B8  | ~~Kopie danych w `DataLoader`~~ **✅ rozwiązane (B8)** | `_select_last` (`@view`), `perm` w konstruktorze (`layers.jl`)                                              | ~~kopia batcha i całego zbioru przy shuffle~~ → widoki + wektor perm (~469 KiB); warianty `_select_last_copy`/`_dataloader_shuffle_copy` do testów |
| B9  | Alokacje w Dropout                           | `forward(dropout_op)`                                                                                      | tymczasowa tablica `rand(T, size(x))` + nowa `BitArray` maski per batch; w trybie eval dodatkowo `copy(x)` |


### 6.4. Rekomendowane kroki (priorytetyzowane)

> **Status**:
> - krok **P0 (im2col + GEMM) zrealizowany** — zob. `OPTYMALIZACJA-P0-IM2COL.md`
>   (czas epoki ~150–190 s → ~16 s, test acc 87.5%),
> - krok **P1 (preałokacja buforów + operacje in-place, B3/B5) zrealizowany** —
>   zob. `OPTYMALIZACJA-P1-PREALOKACJA.md` (alokacje epoki 25.3 GiB → 2.38 GiB,
>   krok treningowy 4.23 MiB → 323 KiB, czas epoki ~14.8 s, identyczna trajektoria
>   uczenia),
> - krok **P1 (akumulacja gradientu in-place, B4) zrealizowany** — zob.
>   `OPTYMALIZACJA-P1-AKUMULACJA.md` (rozłączne bufory gradientów `gradbuf` —
>   aliasing z 6.5 usunięty; protokół jąder `backward!` in-place dla `*`, splotu
>   i MaxPoola; pominięcie gradientu wejść `Constant`; backward 317 KiB → 17 KiB,
>   epoka ~12.1 s / 680 MiB, identyczna trajektoria uczenia),
> - krok **P1 (cache argmax w MaxPool, B6) zrealizowany** — zob.
>   `OPTYMALIZACJA-P1-ARGMAX.md` (zapis `:ihi`/`:iwi` w forward; backward scatter
>   z cache zamiast ponownego skanu okien; backward MaxPool 183 µs → 12 µs,
>   pełny backward! 1.1 ms → 0.7 ms, epoka ~9.7 s szac., identyczna trajektoria),
> - krok **P1 (widoki zamiast kopii w DataLoaderze, B8) zrealizowany** — zob.
>   `OPTYMALIZACJA-P1-DATALOADER.md` (`@view` w `_select_last`; shuffle przez
>   `perm` zamiast kopii zbioru; konstruktor shuffle 182 MiB → 469 KiB,
>   batch 31 KiB → 384 B, alokacje epoki 680 MiB → 324 MiB, identyczna trajektoria),
> - krok **P2 (stabilność typów węzłów grafu, B2) zrealizowany** — zob.
>   `OPTYMALIZACJA-P2-TYPY.md` (inferencja `T,G` w `BroadcastedOperator`;
>   `Array{Float32,4}`/`Matrix{Float32}`/`Float32` zamiast `NodeValue`; widoki
>   `SubArray` normalizowane do `Array`; 0 węzłów `NodeValue` w CNN, identyczna trajektoria).


| Prio   | Krok                                                                                                                                                                                 | Gdzie                                    | Oczekiwany efekt                                              | Nakład      |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- | ------------------------------------------------------------- | ----------- |
| **P0** | Zastąpić splot schematem **im2col + GEMM** (`mul!` na buforze) dla forward i backward; alternatywnie minimalnie: przestawić pętle pod układ column-major i dodać `@simd`             | `_conv_forward`/`_conv_backward` (B1)    | największe przyspieszenie (BLAS), redukcja czasu nawet 10–50× | średni      |
| **P1** | **Preałokować bufory** wyjść i gradientów i pisać in-place (`mul!`, `.=`, `fill!`)                                                                                                   | `_compute!`, `zerograd!` (B3, B5)        | spadek alokacji z ~10 GiB → bliski zeru, mniej GC             | średni      |
| **P1** | Akumulacja gradientu **in-place** (`n.gradient .+= g`) — **najpierw usunąć aliasing** (zob. 6.5): operatory pass-through muszą kopiować lub bufory muszą być rozłączne               | `_accumulate!` (B4)                      | eliminacja alokacji per krawędź grafu                         | mały/średni |
| **P1** | **Cache argmax** w MaxPool (zapis indeksów w forward, użycie w backward)                                                                                                             | `MaxPool` + `_maxpool_*` (B6)            | mniej pracy CPU w backward                                    | mały        |
| **P1** | **Widoki zamiast kopii** w `DataLoader` (`@view`/`selectdim` w `_select_last`; shuffle przez permutację indeksów per batch zamiast kopii całego zbioru)                              | `layers.jl` (B8)                         | −180 MB/epokę + brak kopii per batch                          | mały        |
| **P2** | Naprawić **stabilność typów** węzłów: sparametryzować konkretnym typem tablicy (np. `Array{Float32,N}`) zamiast `NodeValue`                                                          | `structures.jl` (+ konstruktory) (B2)    | mniejszy boxing; głównie **umożliwia** bufory in-place z P1 — sam dispatch na ~20 węzłach/batch jest pomijalny | średni/duży |
| **P2** | **Wielowątkowość** po wymiarze batcha / kanałach (`Threads.@threads`) lub `@simd`/`LoopVectorization.@turbo` w pętlach                                                               | splot, pooling (B7)                      | skalowanie z liczbą rdzeni                                    | mały/średni |
| **P2** | Reużycie listy `graph(...)` i buforów `Constant` w `loss_and_accuracy`; preałokowana maska Dropoutu                                                                                  | notatnik / `layers.jl` (B9)              | mniej alokacji w ewaluacji i treningu                         | mały        |
| **P3** | Opcjonalna „szybka ścieżka" delegująca do `NNlib.conv`/`∇conv` za flagą                                                                                                              | `autodiff.jl`                            | walidacja poprawności i benchmark referencyjny                | mały        |


### 6.5. Niezmienniki (czego nie wolno zmienić)

- **Poprawność gradientów** — po każdej zmianie weryfikować różniczkowaniem numerycznym
(finite differences) na małych tensorach oraz utrzymać wynik **~86.8% test acc** po 3 epokach.
- **Zgodność API** — nazwy i sygnatury (`Chain`, `Conv`, `MaxPool`, `flatten`, `Dense`,
`Dropout`, `logitcrossentropy`, `onehotbatch`/`onecold`, `Descent`/`setup`/`update!`,
`forward!`/`backward!`/`optimize!`) muszą działać bez zmian w notatniku.
- **Współdzielenie pamięci** `Variable` ↔ tablice warstw (warunek działania `optimize!` in-place).
- **Determinizm Dropoutu** — backward musi używać tej samej maski co forward (zob. B-cache).
- **Aliasing gradientów — ✅ ROZWIĄZANE w kroku B4** (`OPTYMALIZACJA-P1-AKUMULACJA.md`).
  Historycznie: operatory pass-through (`identity`, `flatten`, Dropout w eval)
  zwracały w `backward` gradient współdzielący pamięć z gradientem wejściowym,
  a pierwsza akumulacja zapisywała referencję bez kopii — naiwne `.+=` psułoby
  wyniki. Od kroku B4 każdy operator ma **rozłączny, zachowany bufor**
  (`gradbuf`): pierwszy wkład jest kopiowany, kolejne dosumowywane w miejscu;
  jądra in-place piszą wyłącznie do buforów uzyskanych z `_grad_target!`.
  Niezmiennik do utrzymania w przyszłych krokach: gradient węzła nigdy nie
  współdzieli pamięci z innym węzłem ani z buforami `Workspace` warstw.
  Uwaga: wyjścia `forward` operatorów `identity`/`flatten` nadal aliasują
  wejścia (`x`/`reshape(x)`) — to celowe (zero kopii) i bezpieczne, bo wyjścia
  nie są modyfikowane w miejscu poza właścicielem bufora.

### 6.6. Jak weryfikować postęp

- **Wydajność**: `BenchmarkTools.@btime` na pojedynczym `forward!`/`backward!`; `@time`
całej epoki; profilowanie `Profile`/`@profview`; alokacje przez `--track-allocation=user`.
- **Stabilność typów**: `@code_warntype` / `Cthulhu` na `_compute!`, `forward`, `_conv_forward`
(czerwone `Any`/`Union` = cel do usunięcia).
- **Poprawność**: test gradientu numerycznego dla każdego operatora; test regresji
dokładności (3 epoki, próg ≥ ~86% test acc) jako bramka CI.
- **Kolejność prac**: najpierw P0 (im2col — dominujące wąskie gardło), zweryfikować
poprawność i czas; potem P1 (bufory in-place — po usunięciu aliasingu z 6.5,
cache argmax, widoki w DataLoaderze); na końcu P2/P3 (stabilność typów,
równoległość, ścieżka referencyjna `NNlib`).

