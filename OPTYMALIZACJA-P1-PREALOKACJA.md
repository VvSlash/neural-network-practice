# Optymalizacja P1 — preałokacja buforów i operacje in-place

Drugi krok optymalizacji biblioteki `AWIDNN` według planu z
`PORÓWNANIE-IMPLEMENTACJI.md` (sekcja 6.4, priorytet **P1**, wąskie gardła
**B3** i **B5**, częściowo **B9**). Punktem wyjścia („przed") jest stan po
kroku P0 (`OPTYMALIZACJA-P0-IM2COL.md`).

> **Cel**: wyeliminować alokacje tablic w gorącej pętli treningowej —
> preałokować bufory wyjść operatorów i gradientów oraz pisać w miejscu
> (`mul!`, `.=`, `fill!`, `rand!`) — bez zmiany wyników i bez łamania API.

## 1. Zakres zmian

| Plik | Zmiana |
|---|---|
| `packages/AWIDNN/src/structures.jl` | nowy typ `Workspace` (mapa „rola → tablica"); pola `ws::Workspace` w `Conv`, `MaxPool` i `Dropout` |
| `packages/AWIDNN/src/layers.jl` | konstruktory `Conv`/`MaxPool` przekazują `Workspace()` |
| `packages/AWIDNN/src/autodiff.jl` | pomocnicze `_ensure`/`_fit!`; leniwe bufory wyjść w `_compute!` + protokół `forward!`; jądra in-place dla `+`, `*`, `relu`, `σ`, splotu, MaxPool i Dropout; `zerograd!` z `fill!`; akumulacja in-place dla `Variable` |
| `scripts/benchmark_bottlenecks.jl` | sekcja B9 benchmarkuje dodatkowo wariant in-place Dropoutu |

### Idea rozwiązania

1. **Leniwe bufory wyjść (B3)** — `_compute!` przy pierwszym przebiegu woła
   dotychczasowe, alokujące `forward` (ustala kształt), a w kolejnych
   iteracjach `forward!(out, n, ...)`, które pisze do istniejącego bufora
   `n.output`. Pomocnicze `_ensure(out, T, dims)` realokuje bufor wyłącznie
   przy zmianie kształtu/typu (np. inny batchsize). Operatory bez jądra
   in-place korzystają z wariantu domyślnego (alokującego) — np. `flatten`
   (czysty `reshape`, O(1)) i skalarna strata.
2. **Przestrzeń robocza warstwy (`Workspace`)** — tablice pośrednie splotu
   (`:cols`, `:Wm`, `:Y`, `:G`, `:gWm`, `:gcols`) oraz **gradienty wyników**
   (`:gx`, `:gW` splotu, `:gx` MaxPoola) żyją w `ws` obiektu warstwy
   i są reużywane między iteracjami (pozyskiwanie przez `_fit!`).
   Ponieważ warstwa jest współdzielona między grafami, bufory są reużywane
   także między batchami ewaluacji.
3. **Zerowanie gradientów (B5)** — `zerograd!` dla `Variable` wykonuje
   `fill!(gradient, 0)` zamiast alokować `zero(output)` w każdym `backward!`.
4. **Akumulacja in-place dla `Variable`** — `n.gradient .+= g`. Jest to
   bezpieczne, bo bufor gradientu jest własnością wyłączną `Variable`
   (tworzony w `zerograd!`), a `g` pochodzi ze świeżych lub roboczych tablic
   jąder backward — nigdy nie jest tym samym buforem.
5. **Dropout (B9, częściowo)** — losowanie `rand!` do bufora `:rand`,
   maska `BitArray` reużywana w miejscu, wynik do bufora węzła.

### Granica kroku (czego świadomie NIE zmieniono)

Akumulacja gradientów **operatorów** (`_accumulate!(::Operator, ...)`)
pozostała nie-in-place — to osobny krok P1 („akumulacja gradientu in-place"),
który wymaga wcześniejszego usunięcia aliasingu operatorów pass-through
(`identity`/`flatten`/Dropout w trybie eval zwracają `g` lub `reshape(g)` —
zob. ostrzeżenie w sekcji 6.5 dokumentu porównawczego). Z tego samego powodu
alokują nadal jądra backward warstwy `Dense` (`g*Bᵀ`, `Aᵀ*g`) — to one
dominują pozostałe alokacje kroku (zob. sekcja 4).

### Reguła własności buforów (ważne dla kolejnych kroków)

Zwracane przez backward `gx`/`gW` splotu i `gx` MaxPoola wskazują na bufory
`Workspace` i są **nadpisywane przy następnym wywołaniu backward tej samej
warstwy**. Jest to poprawne, bo gradienty są czytane wyłącznie w obrębie
jednego przejścia `backward!` (akumulacja do `Variable` kopiuje wartości),
ale stanowi dodatkowy przypadek aliasingu do uwzględnienia przy
implementacji in-place akumulacji operatorów. Ograniczenie: jedna warstwa
użyta dwukrotnie w grafie (współdzielenie wag) współdzieliłaby bufory —
w obecnej architekturze sieci ten przypadek nie występuje.

## 2. Metodologia

Identyczna jak w P0: `julia scripts/benchmark_bottlenecks.jl` (dane
syntetyczne, rozmiary sieci z notatnika, batch 10, `@btime`), Julia 1.11.5,
1 wątek, Windows. Wartości „przed" pochodzą z przebiegu kończącego krok P0.

## 3. Wyniki — splot i MaxPool (B1/B6: alokacje per wywołanie)

| Operacja | Przed (po P0) | Po (P1) |
|---|---|---|
| conv1 forward | 140.3 µs (12 alok., 643.8 KiB) | 127.4 µs (**4 alok., 183.9 KiB** — wyłącznie tablica wyniku) |
| conv2 forward | 194.9 µs (13 alok., 662.3 KiB) | 190.3 µs (**4 alok., 122.7 KiB**) |
| conv1 backward | 273.6 µs (19 alok., 766.9 KiB) | 253.3 µs (**1 alok., 64 B**) |
| conv2 backward | 448.1 µs (22 alok., 1006.1 KiB) | 423.5 µs (**1 alok., 64 B**) |
| MaxPool backward | 196.3 µs (3 alok., 183.9 KiB) | 184.0 µs (**0 alok., 0 B**) |
| Dropout forward (84×10) | 3.8 µs (9 alok., 7.05 KiB) | 4.7 µs in-place (**512 B**) |

Uwagi: forward przez wrapper `_conv_forward` alokuje już tylko tablicę
wyniku (w grafie i ta znika — wynik trafia do bufora węzła). Wariant in-place
Dropoutu jest przy tym mikro-rozmiarze (84×10) minimalnie wolniejszy
czasowo (narzut rozgłoszenia na `BitArray`), ale alokuje 14× mniej bajtów;
w pełnym kroku różnica czasu jest niemierzalna.

## 4. Wyniki — pełny krok treningowy (B2–B5)

| Pomiar | Przed (po P0) | Po (P1) | Zmiana |
|---|---|---|---|
| `forward!` (cały graf) | 537.2 µs (114 alok., **1.37 MiB**) | 506.6 µs (78 alok., **4.83 KiB**) | pamięć **290×** mniej |
| `backward!` (cały graf) | 1.426 ms (474 alok., **2.86 MiB**) | 1.312 ms (399 alok., **317.3 KiB**) | pamięć **9.2×** mniej |
| pełny krok (fwd+bwd+SGD) | 2.074 ms (606 alok., **4.23 MiB**) | 1.908 ms (495 alok., **322.6 KiB**) | czas −8%, pamięć **13×** mniej |
| szacunkowy czas epoki (6000 batchy) | 12.4 s | 11.4 s | −8% |

Pozostałe ~317 KiB w `backward!` to niemal w całości jądra backward warstwy
`Dense` (`g*Bᵀ` dla wag 84×784 ≈ 263 KiB) oraz drobne tablice operatorów —
domena następnego kroku P1 (akumulacja/jądra backward in-place po usunięciu
aliasingu).

## 5. Profil (`Profile`, odpowiednik tekstowy `@profview`)

Po P1 liczba próbek `backward!` spadła z 63 do 51, a z listy gorących linii
zniknęły alokujące jądra; w czołówce pozostają obliczeniowe części splotu
(`_conv_backward` — GEMM-y i `_col2im!`), MaxPool oraz backward `Dense`:

```
    51  autodiff.jl  137  backward!
    44  autodiff.jl  146  backward!(order::Vector{GraphNode}; seed)
    27  autodiff.jl  538  backward(::BroadcastedOperator{typeof(conv_op)}, ...)
    21  autodiff.jl   84  _compute!(n::BroadcastedOperator{typeof(maxpool_op)})
    21  autodiff.jl   64  forward!(order::Vector{GraphNode})
    12  autodiff.jl  536  forward!(out, ::BroadcastedOperator{typeof(conv_op)}, ...)
    11  autodiff.jl  637  backward(::BroadcastedOperator{typeof(maxpool_op)}, ...)
     6  autodiff.jl  250  backward(::BroadcastedOperator{typeof(*)}, ...)   # Dense
     6  autodiff.jl  426  _im2col!(...)
```

Zakomentowana sekcja `@profview` w `scripts/benchmark_bottlenecks.jl`
pozostaje dostępna do profilowania interaktywnego.

## 6. Testy poprawności

1. **Zgodność z referencją naive** (te same asercje co w P0, wykonywane przy
   każdym uruchomieniu skryptu): forward `max|różnica| = 1.19e-6`; backward
   gx = `2.38e-6`, gW = `1.83e-4` (epsilon Float32) — **bez zmian po P1**,
   mimo że ścieżka używa teraz buforów roboczych.
2. **Gradient numeryczny** (Float64, centralne różnice): `dL/dx` i `dL/dW`
   zgodne do ~9 cyfr znaczących — bez zmian.
3. **Regresja treningu** (`scripts/validate_training.jl`, seed 42) — przebieg
   **identyczny co do wartości** z przebiegiem po P0 (te same liczby losowe,
   ta sama arytmetyka):

```
epoka 1: train acc = 84.88%, test acc = 83.95%
epoka 2: train acc = 87.19%, test acc = 86.04%
epoka 3: train acc = 88.92%, test acc = 87.50%
```

Identyczna trajektoria uczenia potwierdza, że operacje in-place nie zmieniły
wyników nawet na poziomie pojedynczych bitów trajektorii.

## 7. Podsumowanie pełnej epoki treningu

| Pomiar (`@time` epoki, batchsize 10) | Po P0 | Po P1 |
|---|---|---|
| czas epoki (bez kompilacji) | ~16.0 s | **~14.8 s** |
| alokacje pamięci na epokę | 25.3 GiB | **2.38 GiB** (10.6× mniej) |
| udział GC | ~5.7% | **~1.8%** |
| test acc po 3 epokach | 87.5% | 87.5% (identyczna) |

Pozostałe 2.38 GiB na epokę pochodzi głównie z: jąder backward `Dense`
(~1.5 GiB), kopii batchy w `DataLoader` (~190 MB + 188 MB kopii całego
zbioru przy shuffle) oraz grafów ewaluacji budowanych per batch — kolejne
kroki P1 (widoki w DataLoaderze) i P2 (reużycie grafu w ewaluacji).

## 8. Makra testowe i możliwość ponownego badania

- **Wewnątrz biblioteki `AWIDNN` nadal nie ma żadnych makr testowych** —
  benchmarki sięgają do funkcji wewnętrznych z zewnątrz (`AWIDNN._conv_forward`
  itd.).
- Warianty referencyjne `_conv_forward_naive`/`_conv_backward_naive`
  pozostają w bibliotece; ścieżki alokujące (`forward`, `_conv_forward`,
  `_maxpool_forward`) pozostają dostępne obok wariantów in-place, więc
  wąskie gardła B3/B5/B9 można ponownie zbadać tym samym skryptem
  (sekcja B9 porównuje oba warianty automatycznie).
- Makro `@profview` pozostaje **zakomentowane** na końcu
  `scripts/benchmark_bottlenecks.jl`.

## 9. Jak powtórzyć

```bash
# benchmarki wąskich gardeł B1–B9 + testy poprawności
julia scripts/benchmark_bottlenecks.jl

# pełny test regresji treningu (3 epoki, dokładność + czasy + alokacje)
julia scripts/validate_training.jl
```
