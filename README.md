# neural-network-practice

Automatyczne różniczkowanie i sieć neuronowa (CNN) napisana w języku Julia.
Repozytorium zawiera własną bibliotekę `AWIDNN` (`packages/AWIDNN`) oraz dwa notatniki Jupyter:

- `AWID-2026-CNN-AWIDNN.ipynb` – model oparty na własnej bibliotece `AWIDNN`,
- `AWID-2026-CNN-FLUX.ipynb` – referencyjny model oparty na `Flux`.

## Wymagania

- **Julia 1.11** (notatniki przygotowano w 1.11.5).
- Jupyter uruchamiany przez pakiet `IJulia` (instalowany automatycznie poniżej).

## Uruchomienie notatnika

Wszystkie polecenia wykonuj z katalogu głównego repozytorium.

1. **Instalacja środowiska** – pobiera zależności i rejestruje lokalną bibliotekę `AWIDNN`:

```bash
julia scripts/setup_env.jl
```

2. **Rejestracja kernela Jupyter** (kernel `Julia (AWID)` z aktywnym projektem repozytorium):

```bash
julia scripts/install_ijulia_kernel.jl
```

3. **Uruchomienie Jupyter**:

```bash
julia scripts/open_jupyter.jl
```

W przeglądarce otwórz wybrany notatnik i wybierz kernel **`Julia (AWID)`**.

## Uwagi

- Pierwsze uruchomienie pobiera pakiety oraz zbiór danych FashionMNIST i może chwilę potrwać.
- Notatniki same aktywują środowisko repozytorium (`Pkg.activate`), więc kluczowe jest uruchomienie ich na kernelu `Julia (AWID)`.
