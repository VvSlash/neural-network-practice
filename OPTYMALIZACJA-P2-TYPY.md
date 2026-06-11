# Optymalizacja P2/B2 — stabilność typów węzłów grafu

Szósty krok optymalizacji biblioteki `AWIDNN` według planu z
`PORÓWNANIE-IMPLEMENTACJI.md` (sekcja 6.4, priorytet **P2**, wąskie gardło
**B2**). Punktem wyjścia („przed") jest stan po kroku P1/B8
(`OPTYMALIZACJA-P1-DATALOADER.md`).

> **Cel**: zastąpić abstrakcyjny typ `NodeValue` (`Union{Real, AbstractArray}`)
> konkretnymi typami tablic (`Array{Float32,4}`, `Matrix{Float32}`, `Float32`
> dla straty) w parametrach `BroadcastedOperator` — mniej dynamicznego dispatchu
> i boxingu na granicy węzłów — bez zmiany wyników i bez łamania API.

## 1. Zakres zmian

| Plik | Zmiana |
|---|---|
| `packages/AWIDNN/src/structures.jl` | komentarz o inferencji typów; usunięty domyślny konstruktor `NodeValue` |
| `packages/AWIDNN/src/autodiff.jl` | `_valtype_of`, `_infer_op_types`, `_storage_array_type`; konstruktor `BroadcastedOperator` z inferencją; wyspecjalizowane `_compute!` (tablica/skalar/NodeValue); wyspecjalizowane `_accumulate!` dla `Variable{T,T}` i `BroadcastedOperator{F,T,G}` |
| `scripts/benchmark_bottlenecks.jl` | sekcja B2: weryfikacja braku `NodeValue` w grafie CNN + przykładowe typy |

### Idea rozwiązania

1. **Inferencja przy budowie grafu** — konstruktor `BroadcastedOperator` zbiera
   typ pierwszego tablicowego wejścia (`_first_array_valtype`) i przypisuje
   konkretne `T,G`:
   - operatory elementowe / splot / pooling: `Array{Float32,4}` lub `Matrix{Float32}`,
   - `flatten`: `Matrix{eltype(A)}`,
   - `logitcrossentropy` / `sum`: `Float32` (skalarna strata),
   - nieznane wejścia: zapasowy `NodeValue` (zachowanie sprzed B2).
2. **Normalizacja widoków** — `_storage_array_type` mapuje `SubArray{E,N}`
   (batch z DataLoadera B8) na `Array{E,N}`, bo bufory wyjść operatorów są
   zawsze gęstymi tablicami (`_ensure` alokuje `Array`).
3. **Wyspecjalizowane ścieżki** — `_compute!` i `_accumulate!` mają warianty
   dla `T<:AbstractArray`, `T<:Real` oraz zapasowy `T<:NodeValue`; kompilator
   monomorfizuje pętlę grafu po konkretnych typach węzłów CNN.
4. **API bez zmian** — `layers.jl` i notatnik nie wymagają modyfikacji;
   inferencja jest przezroczysta.

## 2. Metodologia

`julia scripts/benchmark_bottlenecks.jl` + `julia scripts/validate_training.jl`,
Julia 1.11.5, 1 wątek, Windows. Weryfikacja typów: brak węzłów
`BroadcastedOperator{…, NodeValue, NodeValue}` w grafie sieci CNN;
profil pokazuje monomorficzne `_compute!` (np. `Array{Float32,4}`).

## 3. Wyniki — stabilność typów (B2)

Graf CNN (25 węzłów, 12 operatorów `BroadcastedOperator`):

```
conv=Array{Float32, 4}, maxpool=Array{Float32, 4}, flatten=Matrix{Float32},
W*x=Matrix{Float32}, logitcrossentropy=Float32, ...
```

Węzły z `NodeValue`: **0** (asercja w skrypcie benchmarków).

Profil (50 kroków) — `_compute!` monomorficzny, np.:

```
_compute!(n::BroadcastedOperator{typeof(conv_op), Array{Float32, 4}, Array{Float32, 4}})
forward!(::BroadcastedOperator{typeof(*), Matrix{Float32}, Matrix{Float32}}, ...)
backward!(::BroadcastedOperator{typeof(conv_op), Array{Float32, 4}, ...}, ...)
```

## 4. Wyniki — pełny krok treningowy (B2–B5)

| Pomiar | Przed (po B8) | Po (B2) | Zmiana |
|---|---|---|---|
| `forward!` (cały graf) | 768 µs (78 alok., 4.81 KiB) | 768 µs (78 alok., 4.81 KiB) | bez istotnej zmiany |
| `backward!` (cały graf) | 787 µs (274 alok., 17.0 KiB) | 787 µs (274 alok., 17.0 KiB) | bez istotnej zmiany |
| pełny krok (fwd+bwd+SGD) | 1.69 ms | 1.69 ms | bez istotnej zmiany |
| szacunkowy czas epoki | 10.0 s | 10.0 s | bez istotnej zmiany |

Zgodnie z prognozą z sekcji 6.3 dokumentu porównawczego: dispatch na ~20
węzłach/batch był pomijalny — główny zysk B2 to **stabilność typów** (mniej
boxingu, monomorficzne `_compute!`), nie skrócenie czasu epoki.

## 5. Testy poprawności

1. **Graf CNN** — `forward!` + `backward!` bez błędu; zero węzłów `NodeValue`.
2. **Gradient numeryczny na pełnym grafie** — max względna różnica `9.81e-9`
   (bez zmian).
3. **Regresja treningu** (seed 42, batchy jako widoki z DataLoadera):

```
epoka 1: train acc = 84.88%, test acc = 83.95%
epoka 2: train acc = 87.19%, test acc = 86.04%
epoka 3: train acc = 88.92%, test acc = 87.50%
```

## 6. Podsumowanie epoki treningu

| Pomiar | Po B8 | Po B2 |
|---|---|---|
| czas epoki (bez kompilacji) | ~12.0 s | **~12.0 s** |
| alokacje epoki | 324 MiB | **323 MiB** |
| test acc | 87.5% | 87.5% (identyczna) |

## 7. Jak powtórzyć

```bash
julia scripts/benchmark_bottlenecks.jl
julia scripts/validate_training.jl
```
