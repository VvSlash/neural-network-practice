# neural-network-practice

Automatyczne różniczkowanie i sieć neuronowa (CNN) napisana w języku Julia.
Repozytorium zawiera własną bibliotekę `AWIDNN` (`packages/AWIDNN`) oraz notatniki Jupyter:

- `AWID-2026-CNN-AWIDNN.ipynb` – model oparty na własnej bibliotece `AWIDNN`,
- `AWID-2026-CNN-FLUX.ipynb` – referencyjny model oparty na `Flux`,
- `AWID-2026-CNN-PYTORCH.ipynb` – bliźniaczy model oparty na `PyTorch` (Python),
- `AWID-2026-CNN-TENSORFLOW.ipynb` – bliźniaczy model oparty na `TensorFlow` (Python),
- `AWID-2026-CNN-KERAS.ipynb` – bliźniaczy model oparty na `Keras` (Python).

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

## Uruchomienie notatników Pythona (PyTorch / TensorFlow / Keras)

Notatniki `AWID-2026-CNN-PYTORCH.ipynb`, `AWID-2026-CNN-TENSORFLOW.ipynb` oraz `AWID-2026-CNN-KERAS.ipynb` to niezależne, bliźniacze wersje modelu napisane w Pythonie (odpowiednio z użyciem `PyTorch`, `TensorFlow` i `Keras`). Wszystkie korzystają z tego samego wirtualnego środowiska Pythona i wspólnego kernela Jupyter (Keras 3 jest instalowany razem z `tensorflow`).

### Wymagania

- **Python 3.12** (notatniki przygotowano i przetestowano w 3.12).
- Pakiety wymienione w `requirements.txt` (m.in. `torch`, `torchvision`, `tensorflow`, `matplotlib`, `jupyter`).

### Instalacja

Wszystkie polecenia wykonuj z katalogu głównego repozytorium.

1. **Utworzenie wirtualnego środowiska** (Python 3.12):

```powershell
py -V:3.12 -m venv .venv
```

> Na systemach Linux/macOS użyj `python3.12 -m venv .venv`.

2. **Aktywacja środowiska**:

```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

3. **Instalacja zależności**:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. **Rejestracja wspólnego kernela Jupyter** (kernel `Python (AWID)`):

```bash
python -m ipykernel install --user --name awid-python --display-name "Python (AWID)"
```

5. **Uruchomienie Jupyter**:

```bash
python -m jupyter notebook
```

W przeglądarce otwórz `AWID-2026-CNN-PYTORCH.ipynb`, `AWID-2026-CNN-TENSORFLOW.ipynb` lub `AWID-2026-CNN-KERAS.ipynb` i wybierz kernel **`Python (AWID)`**.

## Uwagi

- Pierwsze uruchomienie pobiera pakiety oraz zbiór danych FashionMNIST i może chwilę potrwać.
- Notatniki Julia same aktywują środowisko repozytorium (`Pkg.activate`), więc kluczowe jest uruchomienie ich na kernelu `Julia (AWID)`.
- Notatniki Pythona pobierają zbiór FashionMNIST (PyTorch przez `torchvision` do katalogu `data/`, TensorFlow i Keras przez `keras.datasets` do `~/.keras/`); oba katalogi są ignorowane w gicie.
- Trening sieci w PyTorch, TensorFlow i Keras odbywa się na CPU (ok. 13–20 s na epokę); zbiór danych pobierany jest tylko przy pierwszym uruchomieniu.
- Porównanie wszystkich implementacji (różnice i wyniki) znajduje się w `PORÓWNANIE-IMPLEMENTACJI.md`.
