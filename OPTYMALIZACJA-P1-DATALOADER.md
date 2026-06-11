# Optymalizacja P1/B8 — widoki zamiast kopii w DataLoaderze

Piąty krok optymalizacji biblioteki `AWIDNN` według planu z
`PORÓWNANIE-IMPLEMENTACJI.md` (sekcja 6.4, priorytet **P1**, wąskie gardło
**B8**). Punktem wyjścia („przed") jest stan po kroku P1/B6
(`OPTYMALIZACJA-P1-ARGMAX.md`).

> **Cel**: wyeliminować kopie danych w `DataLoader` — batchy jako widoki
> (`@view`) zamiast materializowanych wycinków; tasowanie przez permutację
> indeksów zamiast kopiowania całego zbioru — bez zmiany wyników i bez
> łamania API.

## 1. Zakres zmian

| Plik | Zmiana |
|---|---|
| `packages/AWIDNN/src/structures.jl` | pole `perm::Union{Nothing, Vector{Int}}` w `DataLoader` — wspólna permutacja próbek |
| `packages/AWIDNN/src/layers.jl` | `_select_last` z `@view`; `iterate` korzysta z `perm` per batch; konstruktor bez kopii przy `shuffle`; warianty referencyjne `_select_last_copy`, `_dataloader_shuffle_copy` |
| `scripts/benchmark_bottlenecks.jl` | benchmark B8 przed/po; testy zgodności batcha i shuffle (batche 1, 2, 6000) |

### Idea rozwiązania

1. **Widoki batchy** — `_select_last` zwraca `@view X[colons..., range]`
   zamiast materializowanego wycinka. Dla batcha 10 obrazów 28×28×1
   (~31 KiB) alokacja spada do ~384 B (metadane `SubArray` + krotka).
2. **Tasowanie bez kopii zbioru** — przy `shuffle=true` konstruktor zapisuje
   jedynie wektor `perm = randperm(N)` (~469 KiB dla N = 60 000) zamiast
   kopiować cały tensor (~182 MiB dla X + Y FashionMNIST). Kolejność
   batchy jest ustalana przez `@view perm[state:last_idx]` w `iterate`.
3. **Bezpieczeństwo widoków** — pętla treningowa w notatniku i tak kopiuje
   batch do buforów `Constant` (`input.output .= x`), więc widoki nie są
   modyfikowane przez operatory grafu.
4. **Warianty referencyjne** — `_select_last_copy` i `_dataloader_shuffle_copy`
   zachowane do benchmarków „przed" i testów poprawności.

## 2. Metodologia

Identyczna jak w poprzednich krokach: `julia scripts/benchmark_bottlenecks.jl`
(dane syntetyczne 60 000 × 28×28×1, batch 10, `@btime`), Julia 1.11.5,
1 wątek, Windows. Wartości „przed" z wariantów `_select_last_copy` /
`_dataloader_shuffle_copy` w tym samym przebiegu.

## 3. Wyniki — DataLoader izolowany (B8)

| Pomiar | Przed (kopia) | Po (widok B8) | Zmiana |
|---|---|---|---|
| konstruktor `shuffle=true` | **71–168 ms** (9 alok., **182 MiB**) | **335–575 µs** (3 alok., **469 KiB**) | czas **~200×**, pamięć **~390×** mniej |
| pobranie jednego batcha | **2.7 µs** (5 alok., **31.2 KiB**) | **56–121 ns** (3 alok., **384 B**) | pamięć batcha **~80×** mniej |

Zgodność numeryczna: batch widok vs kopia `max|roznica| = 0.0`; shuffle
perm vs copy — batche 1, 2 i 6000 identyczne.

## 4. Wyniki — pełny krok treningowy (B2–B5)

| Pomiar | Przed (po B6) | Po (B8) | Zmiana |
|---|---|---|---|
| `forward!` (cały graf) | 742–766 µs | 770 µs | bez istotnej zmiany (krok dotyczy DataLoadera) |
| `backward!` (cały graf) | 710–780 µs | 780 µs | bez istotnej zmiany |
| pełny krok (fwd+bwd+SGD) | 1.59–1.68 ms | 1.68 ms | bez istotnej zmiany (benchmark bez DataLoadera) |
| szacunkowy czas epoki (6000 batchy, bez DL) | 9.7–10.0 s | 10.0 s | bez istotnej zmiany |

Główny zysk B8 ujawnia się w **alokacjach epoki z DataLoaderem** (patrz §6),
nie w samym kroku grafu mierzonym w B2–B5.

## 5. Profil (`Profile`, odpowiednik tekstowy `@profview`)

Profil pełnego kroku grafu (50 iteracji) bez zmian względem B6 — `DataLoader`
nie wchodzi w `@profile` tego benchmarku. Zysk B8 jest widoczny w alokacjach
całej epoki (`@time` w `validate_training.jl`).

## 6. Testy poprawności

1. **Batch widok vs kopia** — `_select_last` i `_select_last_copy` dla
   zakresu `1:10` dają identyczne tensory (`max|roznica| = 0`).
2. **Shuffle perm vs copy** — dla tej samej permutacji batche 1, 2 i 6000
   zgadzają się co do wartości.
3. **Regresja treningu** (`scripts/validate_training.jl`, seed 42) —
   trajektoria **identyczna co do wartości**:

```
epoka 1: train acc = 84.88%, test acc = 83.95%
epoka 2: train acc = 87.19%, test acc = 86.04%
epoka 3: train acc = 88.92%, test acc = 87.50%
```

## 7. Podsumowanie pełnej epoki treningu

| Pomiar (`@time` epoki, batchsize 10) | Po B6 | Po B8 |
|---|---|---|
| czas epoki (bez kompilacji) | ~12.2 s | **~12.4 s** (wahanie ±0.5 s) |
| alokacje pamięci na epokę | **680 MiB** | **324 MiB** (**2.1×** mniej) |
| udział GC | ~0.4–1.4% | **~0.04–0.08%** |
| test acc po 3 epokach | 87.5% | 87.5% (identyczna) |

Źródło oszczędności ~356 MiB/epokę: brak kopii całego zbioru przy shuffle
(~182 MiB × 1 na epokę) oraz brak kopii 6000 batchy (~31 KiB × 6000 ≈ 186 MiB).

## 8. Makra testowe i możliwość ponownego badania

- **Wewnątrz biblioteki `AWIDNN` nadal nie ma żadnych makr testowych**;
  benchmarki i testy sięgają do internals z zewnątrz.
- Warianty `_select_last_copy` i `_dataloader_shuffle_copy` pozostają
  w bibliotece — umożliwiają ponowne zbadanie wąskiego gardła B8.
- Zakomentowane makro `@profview` w `scripts/benchmark_bottlenecks.jl`
  pozostaje bez zmian.

## 9. Jak powtórzyć

```bash
# benchmarki B1–B9 + testy poprawności (w tym DataLoader widok vs kopia)
julia scripts/benchmark_bottlenecks.jl

# pełny test regresji treningu (3 epoki, dokładność + czasy + alokacje)
julia scripts/validate_training.jl
```
