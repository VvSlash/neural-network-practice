# Optymalizacja P1/B4 — akumulacja gradientu in-place (usunięcie aliasingu)

Trzeci krok optymalizacji biblioteki `AWIDNN` według planu z
`PORÓWNANIE-IMPLEMENTACJI.md` (sekcja 6.4, priorytet **P1**, wąskie gardło
**B4**). Punktem wyjścia („przed") jest stan po kroku P1-preałokacja
(`OPTYMALIZACJA-P1-PREALOKACJA.md`).

> **Cel**: usunąć aliasing gradientów (warunek z sekcji 6.5 dokumentu
> porównawczego), przejść na akumulację in-place dla operatorów oraz
> wyeliminować pozostałe tablice pośrednie w `backward!` — bez zmiany wyników
> i bez łamania API.

## 1. Zakres zmian

| Plik | Zmiana |
|---|---|
| `packages/AWIDNN/src/structures.jl` | pole `gradbuf` w `BroadcastedOperator` i `ScalarOperator` — zachowany, **rozłączny** bufor gradientu węzła |
| `packages/AWIDNN/src/autodiff.jl` | `_accumulate!` operatorów: kopia-przy-pierwszym-wkładzie + `.+=`; protokół jąder `backward!` in-place (`_grad_target!`, wariant domyślny); jądra in-place dla `*`, splotu (oba warianty) i MaxPoola; `_col2im!`/`_maxpool_backward_into!` z flagą `seeded`; `_unflip_acc!` (bariera funkcyjna) |
| `scripts/benchmark_bottlenecks.jl` | nowy test poprawności: gradient numeryczny na **pełnym grafie** (ścieżka in-place, sieć z biasem konwolucji, Float64) |

### Idea rozwiązania

1. **Rozłączne bufory gradientów (usunięcie aliasingu)** — `_accumulate!`
   dla operatora **kopiuje** pierwszy wkład do zachowanego bufora węzła
   (`gradbuf`, pozyskiwany przez `_ensure`, reużywany między przejściami)
   zamiast zapisywać referencję. Gradient węzła nigdy nie współdzieli pamięci
   z innym węzłem (operatory pass-through `identity`/`flatten`/Dropout-eval
   zwracają `g`/`reshape(g)`) ani z buforem roboczym warstwy. Dzięki temu
   kolejne wkłady można bezpiecznie dosumowywać w miejscu (`.+=`).
   **Ostrzeżenie o aliasingu z sekcji 6.5 jest tym samym rozwiązane.**
2. **Protokół `backward!` in-place** — pętla `backward!` najpierw próbuje
   jądra `backward!(n, wejścia..., g)`, które pisze gradienty **bezpośrednio
   do buforów węzłów-wejść** i zwraca `true`; wariant domyślny zwraca `false`
   i kieruje na ścieżkę klasyczną (`backward` + `_accumulate!`). Pomocnik
   `_grad_target!(inp, T, dims)` zwraca `(bufor, seeded)` — `seeded=false`
   oznacza pierwszy wkład (nadpisanie), `true` — dosumowanie — albo `nothing`
   dla `Constant` (gradient liścia nietrenowalnego jest zbędny).
3. **Jądra in-place**:
   - `*` (Dense): 5-argumentowy `mul!` (`C = α·A·B + β·C`, `β = seeded`) —
     znikają tablice `g*Bᵀ` (263 KiB dla wag 84×784) i `Aᵀ*g`,
   - splot: `dL/dW` przez `_unflip_acc!` wprost do bufora `Variable`,
     `dL/dx` przez `_col2im!(…, seeded)` wprost do bufora węzła wejściowego;
     **gdy wejście jest `Constant`** (obrazy w pierwszej warstwie), gradient
     po `x` jest pomijany w całości — oszczędza to GEMM `gcols` i całe
     `_col2im!`,
   - MaxPool: scatter gradientu wprost do bufora węzła wejściowego
     (`_maxpool_backward_into!`), bez bufora pośredniego i bez kopii.
4. Operatory o małych gradientach (`relu`, `σ`, `+`, Dropout, `flatten`,
   strata) pozostają na ścieżce klasycznej — ich wkłady (~3 KiB) są kopiowane
   do buforów własnych węzłów.

### Pułapka wydajnościowa znaleziona po drodze (bariera funkcyjna)

Pierwsza wersja jądra splotu wykonywała pętlę unflip **bezpośrednio** na
buforze zwróconym przez `_grad_target!`. Bufor ma tam typ abstrakcyjny
(pole `gradient` jest typu `Union{Nothing, NodeValue}`), więc każdy dostęp
do elementu przechodził przez dynamiczny dispatch z boxingiem — pomiar:
**22 659 alokacji i 2.70 ms** na krok (regresja!). Wydzielenie pętli do
osobnej funkcji `_unflip_acc!(Wbuf, gWm, seeded)` (bariera funkcyjna —
specjalizacja po konkretnym typie argumentu) zlikwidowało problem:
**274 alokacje i 1.11 ms**. Wniosek zanotowany dla kroku P2 (stabilność
typów): pętle po tablicach z pól o typach abstrakcyjnych zawsze przez
barierę funkcyjną.

## 2. Metodologia

Identyczna jak w poprzednich krokach: `julia scripts/benchmark_bottlenecks.jl`
(dane syntetyczne, batch 10, `@btime`), Julia 1.11.5, 1 wątek, Windows.
Wartości „przed" pochodzą z przebiegu kończącego krok P1-preałokacja.

## 3. Wyniki — pełny krok treningowy (B2–B5)

| Pomiar | Przed (po P1-preałokacja) | Po (B4) | Zmiana |
|---|---|---|---|
| `forward!` (cały graf) | 506.6 µs (78 alok., 4.83 KiB) | 466–520 µs (78 alok., 4.83 KiB) | bez zmian (krok dotyczy backward) |
| `backward!` (cały graf) | 1.312 ms (399 alok., **317.3 KiB**) | **1.107 ms** (274 alok., **17.1 KiB**) | czas −16%, pamięć **18.5×** mniej |
| pełny krok (fwd+bwd+SGD) | 1.908 ms (495 alok., 322.6 KiB) | **1.702 ms** (370 alok., **22.4 KiB**) | czas −11%, pamięć **14×** mniej |
| szacunkowy czas epoki (6000 batchy) | 11.4 s | **10.1 s** | −11% |

Zysk czasowy pochodzi głównie z pominięcia gradientu `dL/dx` pierwszej
warstwy splotowej (wejście to `Constant` — GEMM `gcols` i `_col2im!` nie są
wykonywane) oraz z eliminacji kopii `g*Bᵀ` w warstwie Dense.

## 4. Profil (`Profile`, odpowiednik tekstowy `@profview`)

Liczba próbek `backward!` spadła z 51 do 39; z czołówki zniknęły alokujące
jądra klasyczne — pozostają obliczeniowe GEMM-y splotu, `_im2col!` i MaxPool:

```
    39  autodiff.jl  137  backward!
    35  autodiff.jl   64  forward!(order::Vector{GraphNode})
    34  autodiff.jl   84  _compute!(n::BroadcastedOperator{typeof(conv_op)})
    20  autodiff.jl  598  forward!(out, ::BroadcastedOperator{typeof(conv_op)}, ...)
    18  autodiff.jl  653  backward!(n::BroadcastedOperator{typeof(conv_op)}, ...)
    13  autodiff.jl  544  _conv_forward!(...)
     8  autodiff.jl  299  forward!(out, ::BroadcastedOperator{typeof(*)}, ...)
     7  autodiff.jl  647  _conv_backward_into!(...)
     6  autodiff.jl  487  _im2col!(...)
     6  autodiff.jl  778  backward!(n::BroadcastedOperator{typeof(maxpool_op)}, ...)
```

Zakomentowana sekcja `@profview` w `scripts/benchmark_bottlenecks.jl`
pozostaje dostępna do profilowania interaktywnego.

## 5. Testy poprawności

1. **Nowy test: gradient numeryczny na pełnym grafie** (ścieżka in-place
   `backward!` + akumulacja; sieć Conv(z biasem)→MaxPool→flatten→Dense→Dense
   w Float64, bez Dropoutu — losowa maska uniemożliwia różnice skończone).
   Porównanie pochodnej centralnej z gradientem `Variable` dla losowego
   elementu **każdego** parametru (wagi i biasy obu typów warstw):

```
W        analityczny = -0.23178109,   numeryczny = -0.23178109
Conv.W   analityczny =  0.19515453,   numeryczny =  0.19515453
Conv.b   analityczny =  0.17095113,   numeryczny =  0.17095113
b        analityczny =  0.051608488,  numeryczny =  0.051608488
OK - max wzgledna roznica = 9.81e-9
```

2. **Dotychczasowe testy** (zgodność z naive, gradient numeryczny jądra
   splotu) — wyniki identyczne jak w P0/P1.
3. **Regresja treningu** (`scripts/validate_training.jl`, seed 42) —
   trajektoria **identyczna co do wartości** z przebiegami po P0 i P1
   (te same operacje arytmetyczne w tej samej kolejności):

```
epoka 1: train acc = 84.88%, test acc = 83.95%
epoka 2: train acc = 87.19%, test acc = 86.04%
epoka 3: train acc = 88.92%, test acc = 87.50%
```

## 6. Podsumowanie pełnej epoki treningu

| Pomiar (`@time` epoki, batchsize 10) | Po P1-preałokacja | Po B4 |
|---|---|---|
| czas epoki (bez kompilacji) | ~14.8 s | **~12.1 s** |
| alokacje pamięci na epokę | 2.38 GiB | **680 MiB** (3.5× mniej) |
| udział GC | ~1.8% | **~0.4–1.4%** |
| test acc po 3 epokach | 87.5% | 87.5% (identyczna) |

Dla porównania z punktem startowym całego programu optymalizacji: epoka
spadła z ~150–190 s (wersja bazowa) do ~12.1 s, a alokacje z ~10 GiB do
680 MiB. Pozostałe alokacje to głównie kopie batchy w `DataLoader`
(~190 MB + 188 MB przy shuffle; krok „widoki zamiast kopii") oraz grafy
ewaluacji budowane per batch (krok P2).

## 7. Makra testowe i możliwość ponownego badania

- **Wewnątrz biblioteki `AWIDNN` nadal nie ma żadnych makr testowych**;
  benchmarki i testy sięgają do internals z zewnątrz.
- Ścieżka klasyczna (`backward` + `_accumulate!`) pozostaje w pełni sprawna
  obok jąder in-place (wariant domyślny protokołu) — wąskie gardło można
  ponownie zbadać wyłączając jądra in-place (wystarczy tymczasowo zmienić
  ich zwrot na `false`).
- Warianty `_naive` splotu oraz **zakomentowane** makro `@profview`
  w `scripts/benchmark_bottlenecks.jl` pozostają bez zmian.

## 8. Jak powtórzyć

```bash
# benchmarki B1–B9 + testy poprawności (w tym gradient pełnego grafu)
julia scripts/benchmark_bottlenecks.jl

# pełny test regresji treningu (3 epoki, dokładność + czasy + alokacje)
julia scripts/validate_training.jl
```
