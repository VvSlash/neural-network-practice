# Optymalizacja P0 — splot im2col + GEMM

Pierwszy krok optymalizacji biblioteki `AWIDNN` według planu z
`PORÓWNANIE-IMPLEMENTACJI.md` (sekcja 6.4, priorytet **P0**, wąskie gardło **B1**).

> **Cel**: zastąpić naiwny splot na 7 zagnieżdżonych pętlach schematem
> **im2col + GEMM** (BLAS) w `_conv_forward`/`_conv_backward`, bez zmiany wyników
> treningu i bez łamania API.

## 1. Zakres zmian

| Plik | Zmiana |
|---|---|
| `packages/AWIDNN/src/autodiff.jl` | nowe `_conv_forward`/`_conv_backward` (im2col + GEMM) + pomocnicze `_im2col!`, `_col2im!`, `_kernel2mat`; import `LinearAlgebra: mul!` |
| `packages/AWIDNN/src/autodiff.jl` | stare implementacje **zachowane** jako `_conv_forward_naive`/`_conv_backward_naive` (referencja do testów poprawności i benchmarków „przed/po") |
| `scripts/benchmark_bottlenecks.jl` | **nowy plik testowy** — benchmarki wąskich gardeł B1–B9 + testy poprawności |
| `scripts/validate_training.jl` | **nowy** test regresji — pełny trening 3 epok jak w notatniku |
| `scripts/add_benchmarktools.jl` | **nowy** — instalacja `BenchmarkTools` i naprawa ścieżki `Pkg.develop` (wskazywała na nieistniejący katalog `AWID-KM1`) |
| `Project.toml` (root) | dodany `BenchmarkTools` (tylko środowisko repozytorium — **nie** biblioteka) |

Operatory grafu (`forward`/`backward` dla `conv_op`) wywołują nowe wersje —
nazwy `_conv_forward`/`_conv_backward` pozostały bez zmian, więc żaden inny kod
nie wymagał modyfikacji.

### Idea algorytmu

1. `_im2col!` — rozkłada otoczenia (receptive fields) wejścia do macierzy
   `cols (K, N)`, gdzie `K = kH·kW·Cin`, `N = H_out·W_out·B`; kolumna = jedno
   okno splotu, padding = zera.
2. `_kernel2mat` — spłaszcza jądro **z flipem** do macierzy `(K, Cout)`
   (zachowana konwencja matematyczna splotu z wariantu naive, zgodna z `Flux.Conv`).
3. Forward: `Y = colsᵀ · Wm` — jedno mnożenie macierzowe (`mul!`, BLAS) zamiast
   7 pętli; wynik wraca do układu `(H_out, W_out, C_out, B)` przez `permutedims`.
4. Backward: dwa GEMM-y — `gWm = cols · G` (gradient jądra, potem unflip) oraz
   `gcols = Wm · Gᵀ`, rozrzucane do `gx` przez `_col2im!` (scatter-add).

Linearyzacje indeksów wierszy/kolumn są zgodne z układem column-major Julii
(`kh` i `h` najszybciej zmienne), więc `_im2col!`/`_col2im!` piszą po pamięci
sekwencyjnie — w przeciwieństwie do naiwnych pętli (zob. B1 w sekcji 6.3
dokumentu porównawczego).

## 2. Metodologia testów

- **Skrypt**: `julia scripts/benchmark_bottlenecks.jl` (uruchamiany z katalogu
  głównego repozytorium). Skrypt pokrywa wszystkie wąskie gardła z sekcji 6.3:
  B1 (splot), B2–B5 (pełny krok treningowy na grafie: czas + alokacje),
  B6 (MaxPool), B7 (liczba wątków w nagłówku), B8 (DataLoader), B9 (Dropout),
  oraz profil `Profile` pełnego kroku treningowego (tekstowy odpowiednik
  `@profview`).
- **Dane syntetyczne** o rozmiarach identycznych z siecią z notatnika
  (batchsize = 10, conv1: 28×28×1×10 ⊛ 3×3×1×6, conv2: 14×14×6×10 ⊛ 3×3×6×16),
  `Random.seed!(1)` dla powtarzalności.
- **Środowisko**: Julia 1.11.5, 1 wątek Julii (BLAS wielowątkowy domyślnie),
  Windows; `BenchmarkTools.@btime` (minimum z wielu próbek).
- Po optymalizacji ten sam skrypt automatycznie benchmarkuje dodatkowo warianty
  referencyjne `_naive` i wykonuje testy poprawności.

## 3. Wyniki — wąskie gardło B1 (splot)

### BenchmarkTools, przed → po

| Operacja | Przed (naive) | Po (im2col+GEMM) | Przyspieszenie |
|---|---|---|---|
| conv1 forward (28×28×1×10 ⊛ 3×3×1×6) | 373.4 µs (3 alok., 183.9 KiB) | **140.3 µs** (12 alok., 643.8 KiB) | **2.7×** |
| conv2 forward (14×14×6×10 ⊛ 3×3×6×16) | 1.171 ms (3 alok., 122.6 KiB) | **194.9 µs** (13 alok., 662.3 KiB) | **6.0×** |
| conv1 backward | 923.2 µs (5 alok., 31.0 KiB) | **273.6 µs** (19 alok., 766.9 KiB) | **3.4×** |
| conv2 backward | 3.508 ms (6 alok., 49.5 KiB) | **448.1 µs** (22 alok., 1006.1 KiB) | **7.8×** |

Surowe wyjście `@btime` (przebieg „po", zawiera oba warianty z tej samej maszyny
i tego samego uruchomienia):

```
[B1] Splot: _conv_forward / _conv_backward
  conv1 forward  (28x28x1x10  * 3x3x1x6):     140.300 μs (12 allocations: 643.75 KiB)
  conv2 forward  (14x14x6x10  * 3x3x6x16):    194.900 μs (13 allocations: 662.25 KiB)
  conv1 backward:                              273.600 μs (19 allocations: 766.92 KiB)
  conv2 backward:                              448.100 μs (22 allocations: 1006.14 KiB)
  --- wariant referencyjny (naiwne pętle, przed P0) ---
  conv1 forward  (naive):                      440.100 μs (3 allocations: 183.85 KiB)
  conv2 forward  (naive):                      1.583 ms (3 allocations: 122.60 KiB)
  conv1 backward (naive):                      924.700 μs (5 allocations: 31.02 KiB)
  conv2 backward (naive):                      3.516 ms (6 allocations: 49.51 KiB)
```

> Alokacje pojedynczego wywołania **wzrosły** (bufory `cols`, `Wm`, `Y`, `G`,
> `gcols` tworzone przy każdym wywołaniu) — to świadomy kompromis kroku P0;
> ich preałokacja to osobny krok **P1** (zob. 6.4 dokumentu porównawczego).

## 4. Wyniki — pełny krok treningowy (B2–B5)

| Pomiar | Przed | Po | Przyspieszenie |
|---|---|---|---|
| `forward!` (cały graf) | 1.805 ms (95 alok., 405 KiB) | **537.2 µs** (114 alok., 1.37 MiB) | **3.4×** |
| `backward!` (cały graf) | 5.200 ms (444 alok., 1.21 MiB) | **1.426 ms** (474 alok., 2.86 MiB) | **3.6×** |
| pełny krok (fwd+bwd+SGD) | 7.147 ms (557 alok., 1.60 MiB) | **2.074 ms** (606 alok., 4.23 MiB) | **3.4×** |
| szacunkowy czas epoki (6000 batchy, sam krok) | 42.9 s | **12.4 s** | **3.5×** |

## 5. Profil (`Profile`, odpowiednik tekstowy `@profview`)

Top wpisy z plików `AWIDNN` dla 50 pełnych kroków treningowych (kolumna 1 =
liczba próbek; im więcej, tym więcej czasu w danej linii).

**Przed** — dominuje splot naiwny (`_conv_backward` linie 316–322,
`_conv_forward` linia 295; `backward` conv = 118 z 164 próbek `backward!`):

```
   164  autodiff.jl   97  backward!
   154  autodiff.jl  106  backward!(order::Vector{GraphNode}; seed)
   118  autodiff.jl  337  backward(::BroadcastedOperator{typeof(conv_op)}, ...)
    66  autodiff.jl   43  forward!(order::Vector{GraphNode})
    65  autodiff.jl   58  _compute!(n::BroadcastedOperator{typeof(conv_op)})
    49  autodiff.jl  335  forward(::BroadcastedOperator{typeof(conv_op)}, ...)
    44  autodiff.jl  320  _conv_backward(...)   # naiwna pętla: gx[...] += go * W[...]
    36  autodiff.jl  319  _conv_backward(...)   # naiwna pętla: gW[...] += go * x[...]
    19  autodiff.jl  424  backward(::BroadcastedOperator{typeof(maxpool_op)}, ...)
    17  autodiff.jl  196  backward(::BroadcastedOperator{typeof(*)}, ...)
    16  autodiff.jl  316  _conv_backward(...)   # naiwna pętla: indeksy okna
    12  autodiff.jl  322  _conv_backward(...)
    12  autodiff.jl  295  _conv_forward(...)    # naiwna pętla forwarda
```

**Po** — liczba próbek splotu spada ~2.5×, a wewnątrz splotu czas przenosi się
do `_im2col!`/GEMM; relatywnie rośnie udział MaxPool (cel kroku P1/P2):

```
    63  autodiff.jl  100  backward!
    57  autodiff.jl  109  backward!(order::Vector{GraphNode}; seed)
    46  autodiff.jl  454  backward(::BroadcastedOperator{typeof(conv_op)}, ...)
    30  autodiff.jl   61  _compute!(n::BroadcastedOperator{typeof(maxpool_op)})
    30  autodiff.jl   46  forward!(order::Vector{GraphNode})
    20  autodiff.jl  452  forward(::BroadcastedOperator{typeof(conv_op)}, ...)
    13  autodiff.jl  354  _im2col!(...)         # rozkład okien (nowy koszt główny splotu)
    12  autodiff.jl  425  _conv_backward(...)   # permutedims! gradientu
    11  autodiff.jl  430  _conv_backward(...)   # GEMM gWm
    10  autodiff.jl  407  _conv_forward(...)    # GEMM Y
```

Interaktywny flamegraph: w `scripts/benchmark_bottlenecks.jl` na końcu znajduje
się **zakomentowana** sekcja `@profview` (do uruchomienia w REPL-u VS Code) —
pozostawiona zgodnie z zasadą „nie usuwać, tylko zakomentować".

## 6. Testy poprawności

Wykonywane automatycznie przez `scripts/benchmark_bottlenecks.jl` po wykryciu
wariantów `_naive`:

1. **Zgodność z referencją** (rozmiary conv2 z sieci, Float32):
   - forward: `max|różnica| = 1.19e-6`,
   - backward: `max|różnica|` gx = `2.38e-6`, gW = `1.83e-4`
     (różnice na poziomie epsilonu Float32 — zmieniona kolejność sumowania w GEMM).
2. **Gradient numeryczny** (centralne różnice, Float64, mały tensor 5×5×2×2):
   - `dL/dx`: analityczny `16.060733696`, numeryczny `16.060733692`,
   - `dL/dW`: analityczny `-2.899770156`, numeryczny `-2.899770156`.
3. **Regresja treningu** (`scripts/validate_training.jl` — pełny trening jak
   w notatniku: 3 epoki, batchsize = 10, SGD η = 1e-2, seed 42):

```
epoka 1: 23.8 s (w tym ~30% kompilacja)   train acc = 84.88%, test acc = 83.95%
epoka 2: 16.0 s                            train acc = 87.19%, test acc = 86.04%
epoka 3: 15.8 s                            train acc = 88.92%, test acc = 87.50%
```

Dokładność testowa **87.5% ≥ 86%** — kryterium z sekcji 6.5 dokumentu
porównawczego spełnione.

## 7. Podsumowanie czasu epoki

| Pomiar | Przed | Po |
|---|---|---|
| pełna epoka treningu (pomiar w notatniku / `validate_training.jl`) | ~150–190 s | **~16 s** (~**10×**) |
| szacunek z kroku treningowego (`@belapsed` × 6000, bez DataLoadera) | 42.9 s | 12.4 s |

> Czas „przed" z notatnika pochodzi z wcześniejszego uruchomienia (inne warunki
> maszyny — m.in. silniejsza presja GC przy pełnych danych); rzetelne porównanie
> mikro to sekcje 3–4, gdzie oba warianty zmierzono w jednym przebiegu.
> Alokacje pełnej epoki **wzrosły** (~25 GiB vs ~10 GiB) przez bufory im2col
> tworzone per wywołanie — ich eliminacja (preałokacja, operacje in-place) to
> krok **P1**.

## 8. Makra testowe i możliwość ponownego badania

- **Wewnątrz biblioteki `AWIDNN` nie umieszczono żadnych makr testowych** —
  benchmarki sięgają do funkcji wewnętrznych z zewnątrz przez kwalifikowane
  nazwy (`AWIDNN._conv_forward` itd.), więc biblioteka pozostaje czysta.
- Zamiast tego w bibliotece **zachowano implementacje referencyjne**
  `_conv_forward_naive`/`_conv_backward_naive` — dzięki nim wąskie gardło B1
  można w przyszłości ponownie zbadać tym samym skryptem (porównanie „przed/po"
  odtwarza się automatycznie przy każdym uruchomieniu).
- Makro `@profview` znajduje się w `scripts/benchmark_bottlenecks.jl` w formie
  **zakomentowanej** sekcji na końcu pliku — odkomentować, aby powtórzyć
  profilowanie interaktywne.

## 9. Jak powtórzyć

```bash
# benchmarki wąskich gardeł B1–B9 + testy poprawności (przed/po)
julia scripts/benchmark_bottlenecks.jl

# pełny test regresji treningu (3 epoki, dokładność + czasy)
julia scripts/validate_training.jl
```
