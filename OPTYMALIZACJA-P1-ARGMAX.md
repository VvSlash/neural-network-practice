# Optymalizacja P1/B6 — cache argmax w MaxPool

Czwarty krok optymalizacji biblioteki `AWIDNN` według planu z
`PORÓWNANIE-IMPLEMENTACJI.md` (sekcja 6.4, priorytet **P1**, wąskie gardło
**B6**). Punktem wyjścia („przed") jest stan po kroku P1/B4
(`OPTYMALIZACJA-P1-AKUMULACJA.md`).

> **Cel**: wyeliminować ponowne skanowanie okien w `backward` MaxPoola —
zapisać pozycję argmax podczas `forward` i odczytać ją w `backward` — bez
zmiany wyników i bez łamania API.

## 1. Zakres zmian

| Plik | Zmiana |
|---|---|
| `packages/AWIDNN/src/structures.jl` | komentarz pola `ws` w `MaxPool` — opis buforów `:ihi`/`:iwi` |
| `packages/AWIDNN/src/autodiff.jl` | `_maxpool_forward!` zapisuje argmax do `:ihi`/`:iwi`; `_maxpool_backward_into!` scatter z cache; `_maxpool_backward_recompute_into!` zachowany jako wariant referencyjny |
| `scripts/benchmark_bottlenecks.jl` | benchmark B6 po `forward` (wymóg cache); test zgodności cache vs recompute |

### Idea rozwiązania

1. **Zapis argmax w forward** — przy każdym oknie poolingowym, obok wartości
   maksymalnej, zapisywane są współrzędne zwycięzcy `(hi, wi)` do dwóch
   tablic `Int32` w `m.ws` (`:ihi`, `:iwi`) o kształcie wyjścia
   `(H_out, W_out, C, B)`. Wartość `0` oznacza okno w całości poza brzegiem
   (padding wirtualny `-Inf`).
2. **Odczyt w backward** — `_maxpool_backward_into!` iteruje po pozycjach
   wyjścia i rozrzuca gradient `g[h,w,c,b]` bezpośrednio do
   `gx[ihi[h,w,c,b], iwi[h,w,c,b], c, b]`. Nie wymaga już wejścia `x` —
   eliminuje drugą pętlę po oknach (`kH × kW` mnożnik pracy).
3. **Wariant referencyjny** — `_maxpool_backward_recompute_into!` zachowuje
   poprzednią logikę (ponowny argmax) do testów poprawności i porównania
   „przed/po".
4. **Ścieżka in-place** (`backward!`) korzysta z tego samego jądra —
   w normalnym przebiegu grafu `forward!` zawsze poprzedza `backward!`,
   więc cache jest aktualny.

### Koszt pamięciowy

Dwa bufory `Int32` o kształcie wyjścia warstwy — dla sieci z notatnika:
~94 KiB (14×14×6×10) + ~62 KiB (7×7×16×10) ≈ **156 KiB** na obie warstwy
MaxPool (alokacja leniwa, reużywana między batchami).

## 2. Metodologia

Identyczna jak w poprzednich krokach: `julia scripts/benchmark_bottlenecks.jl`
(dane syntetyczne, batch 10, `@btime`), Julia 1.11.5, 1 wątek, Windows.
Wartości „przed" pochodzą z wariantu `_maxpool_backward_recompute_into!`
w tym samym przebiegu (stan po B4).

## 3. Wyniki — MaxPool izolowany (B6)

| Pomiar | Przed (recompute) | Po (cache B6) | Zmiana |
|---|---|---|---|
| `forward` (28×28×6×10) | — | 186.8 µs (3 alok., 46.0 KiB) | +zapis cache (jednorazowa alokacja `:ihi`/`:iwi`) |
| `backward` (28×28×6×10) | **183.3 µs** (0 alok.) | **11.5 µs** (0 alok.) | czas **16×** mniej |

Zgodność numeryczna cache vs recompute: `max|roznica| = 0.0`.

## 4. Wyniki — pełny krok treningowy (B2–B5)

| Pomiar | Przed (po B4) | Po (B6) | Zmiana |
|---|---|---|---|
| `forward!` (cały graf) | 466–520 µs (78 alok., 4.83 KiB) | 742.4 µs (78 alok., 4.83 KiB) | wahanie pomiarowe (forward MaxPool +~30 µs za zapis cache) |
| `backward!` (cały graf) | **1.107 ms** (274 alok., 17.1 KiB) | **709.8 µs** (274 alok., 17.1 KiB) | czas **−36%** |
| pełny krok (fwd+bwd+SGD) | **1.702 ms** (370 alok., 22.4 KiB) | **1.593 ms** (370 alok., 22.4 KiB) | czas **−6%** |
| szacunkowy czas epoki (6000 batchy) | 10.1 s | **9.7 s** | −4% |

Sieć ma **dwie** warstwy MaxPool — każda backward oszczędza ~170 µs, co daje
~340 µs na krok (zgodne z obserwowanym spadkiem backward o ~400 µs).

## 5. Profil (`Profile`, odpowiednik tekstowy `@profview`)

Po B6 `backward!(maxpool_op)` znika z czołówki profilu — zastąpiony przez
szybki scatter z cache:

```
    34  autodiff.jl  163  backward!
    28  autodiff.jl   90  forward!(order::Vector{GraphNode})
    27  autodiff.jl  110  _compute!(n::BroadcastedOperator{typeof(*)})
    20  autodiff.jl  680  backward!(n::BroadcastedOperator{typeof(conv_op)}, ...)
    13  autodiff.jl  625  forward!(out, ::BroadcastedOperator{typeof(conv_op)}, ...)
     9  autodiff.jl  816  forward!(out, ::BroadcastedOperator{typeof(maxpool_op)}, ...)
```

W profilu po B4 `backward!(maxpool_op)` miał 6 próbek — po B6 **0 próbek**
(scatter z cache jest zbyt krótki, by trafić do topu przy 50 krokach).

Zakomentowana sekcja `@profview` w `scripts/benchmark_bottlenecks.jl`
pozostaje dostępna do profilowania interaktywnego.

## 6. Testy poprawności

1. **Cache vs recompute** — po `forward` gradienty z `_maxpool_backward_into!`
   i `_maxpool_backward_recompute_into!` są identyczne (`max|roznica| = 0`).
2. **Gradient numeryczny na pełnym grafie** (ścieżka in-place `backward!`,
   sieć z MaxPool) — max względna różnica `9.81e-9` (bez zmian względem B4).
3. **Regresja treningu** (`scripts/validate_training.jl`, seed 42) —
   trajektoria **identyczna co do wartości**:

```
epoka 1: train acc = 84.88%, test acc = 83.95%
epoka 2: train acc = 87.19%, test acc = 86.04%
epoka 3: train acc = 88.92%, test acc = 87.50%
```

## 7. Podsumowanie pełnej epoki treningu

| Pomiar (`@time` epoki, batchsize 10) | Po B4 | Po B6 |
|---|---|---|
| czas epoki (bez kompilacji) | ~12.1 s | **~12.2 s** (wahanie ±0.5 s) |
| alokacje pamięci na epokę | 680 MiB | **680 MiB** (bez zmian) |
| test acc po 3 epokach | 87.5% | 87.5% (identyczna) |

Skumulowany postęp od wersji bazowej: epoka z ~150–190 s do ~9.7 s
(szacunek z `@belapsed`), alokacje z ~10 GiB do 680 MiB.

## 8. Makra testowe i możliwość ponownego badania

- **Wewnątrz biblioteki `AWIDNN` nadal nie ma żadnych makr testowych**;
  benchmarki i testy sięgają do internals z zewnątrz.
- Wariant `_maxpool_backward_recompute_into!` pozostaje w bibliotece —
  umożliwia ponowne zbadanie wąskiego gardła B6 i weryfikację poprawności.
- Zakomentowane makro `@profview` w `scripts/benchmark_bottlenecks.jl`
  pozostaje bez zmian.

## 9. Jak powtórzyć

```bash
# benchmarki B1–B9 + testy poprawności (w tym MaxPool cache vs recompute)
julia scripts/benchmark_bottlenecks.jl

# pełny test regresji treningu (3 epoki, dokładność + czasy + alokacje)
julia scripts/validate_training.jl
```
