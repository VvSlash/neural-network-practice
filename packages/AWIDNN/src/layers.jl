# src/layers.jl
# Konstruktory warstw i funkcje pomocnicze:
# - parametry warstw (Dense/Conv/MaxPool/...),
# - warstwy mapowane na węzły grafu AD ("GraphNode"),
# - funkcje pomocnicze kompatybilne stylistycznie z Flux ("onehotbatch", "onecold", "Descent", "setup", "update!", "DataLoader").

using Random: randn, randperm

# Dense - konstruktor i mapowanie na graf AD

# Warstwa w pełni połączona ("y = σ.(W*x .+ b)").

# Inicjalizacja:
# - "W ~ N(0, 1/sqrt(in))" (inicjalizacja wag z rozkładu normalnego o średniej 0 i odchyleniu standardowym 1/sqrt(in)),
# - "b = 0" o kształcie "out x 1" (kolumnowy bias zgodny z konwencją "features x batch"),
# - typ liczbowy domyślnie "Float32".

function Dense(pair::Pair{<:Integer,<:Integer}, σ = identity; T::Type{<:Real} = Float32)
    fin, fout = pair # wejście/wyjście warstwy: "in => out"
    w = randn(T, fout, fin) ./ sqrt(T(fin)) # W: (out, in), skala 1/sqrt(in) ogranicza eksplozję wariancji
    b = zeros(T, fout, 1) # b: (out, 1), broadcast po osi batcha
    return Dense{T, typeof(w), typeof(b), typeof(σ)}(w, b, σ) # jawna parametryzacja typu daje stabilną inferencję
end

# Funkcja (d::Dense)(x::GraphNode) buduje fragment grafu AD odpowiadający:
# "y = σ(W*x .+ b)".

# "Variable(d.weight)" i "Variable(d.bias)" dzielą pamięć z warstwą.
# Dzięki temu "optimize!" aktualizuje te same tablice, które trzyma obiekt "Dense".
function (d::Dense)(x::GraphNode)
    W = Variable(d.weight; name="W") # macierz wag (współdzielona pamięć, parametr trenowalny)
    b = Variable(d.bias;   name="b") # bias (współdzielona pamięć, parametr trenowalny)
    mul = BroadcastedOperator(*, W, x; name="W*x") # węzeł mnożenia macierzowego
    add = BroadcastedOperator(+, mul, b; name="W*x+b") # dodanie biasu (broadcast po osi batcha)
    if d.σ === identity # bez aktywacji nie tworzy dodatkowego węzła
        return add # affine layer (bez aktywacji)
    else
        return BroadcastedOperator(d.σ, add; name="σ") # aktywacja jako osobny operator w grafie
    end
end

# Chain - składanie warstw

# Funkcja (c::Chain)(x::GraphNode) złoży kolejne warstwy: "(c::Chain)(x) = layer_n(…layer_1(x)…)"
function (c::Chain)(x::GraphNode)
    out = x # akumulator przepływu danych przez kolejne warstwy
    for layer in c.layers
        out = layer(out) # każda warstwa zwraca kolejny "GraphNode"
    end
    return out # ostatni węzeł = wyjście całego modelu
end

# Wrappery funkcji straty na operatory z "autodiff.jl"

# Binary cross-entropy dla każdego elementu, zwraca węzeł o kształcie "size(yhat)".
# Aby dostać skalar: "sum_node(bce(yhat, y))".
bce(yhat::GraphNode, y::GraphNode) = BroadcastedOperator(bce_el, yhat, y; name="bce") # BCE dla każdego elementu; redukcja opcjonalnie osobnym węzłem

# Sumowanie - zwraca węzeł skalarny (przydaje się na końcu grafu, jako "L").
sum_node(x::GraphNode) = BroadcastedOperator(sum, x; name="sum") # jawny operator redukcji do skalarnej straty

# Logit (surowe wyniki modelu dla klasy) cross-entropy - odpowiednik "Flux.logitcrossentropy" (skalar).
logitcrossentropy(yhat::GraphNode, y::GraphNode) =
    BroadcastedOperator(logitcrossentropy, yhat, y; name="logitcrossentropy")        # stabilna CE z logitów (bez osobnego softmax w modelu)

# Pomocnicze odpowiedniki Flux (etykiety, predykcje)

# One-hot (wektor etykiet) encoding etykiet: "labels" (wektor) -> macierz "C x B" typu "Float32".
function onehotbatch(labels, classes)
    C = length(classes) # liczba klas
    B = length(labels) # liczba próbek (batch lub cały zbiór)
    Y = zeros(Float32, C, B) # wynik: kolumny = próbki, wiersze = klasy
    idx = Dict(c => i for (i, c) in enumerate(classes)) # mapowanie "etykieta -> numer wiersza" (0(1) lookup)
    for (j, l) in enumerate(labels)
        Y[idx[l], j] = 1 # ustawia "1" dla klasy poprawnej, resztę zostawia jako 0
    end
    return Y # macierz one-hot kompatybilna z logitcrossentropy (jedna kolumna = jedna próbka, jeden wiersz = jedna klasa)
end

# Zwraca etykiety o najwyższej wartości z każdej kolumny (dla macierzy "C x B").
function onecold(yhat::AbstractMatrix, classes = 1:size(yhat, 1))
    [classes[argmax(view(yhat, :, j))] for j in 1:size(yhat, 2)] # argmax per kolumna (próbka) -> etykieta z przestrzeni "classes"
end
onecold(yhat::AbstractVector, classes = 1:length(yhat)) = classes[argmax(yhat)] # dla pojedynczej próbki (wektor wyników modelu dla klasy) - zwraca etykietę o najwyższej wartości

# Optymalizator

struct Descent
    eta::Float32 # learning rate; Float32 dla stabilnej reprezentacji wartości z API
end

setup(opt::Descent, order::Vector{GraphNode}) = (opt, order) # minimalny "stan optymalizatora": (hiperparametry, graf)

function update!(state::Tuple{Descent, Vector{GraphNode}}, _model = nothing, _grads = nothing)
    opt, order = state # rozpakowanie stanu
    optimize!(order, opt.eta) # faktyczny krok SGD (zaimplementowany w "autodiff.jl")
    return nothing
end

# Placeholder (na przyszłe rozszerzenia API)
_todo(name) = error("AWIDNN.$name: jeszcze nie zaimplementowane - następny krok rozwoju biblioteki.")

# Conv - konstruktor i mapowanie na graf AD
function Conv(kernel::Tuple{Int,Int}, pair::Pair{<:Integer,<:Integer}; pad::Union{Integer, NTuple{2,Integer}} = 0, stride::Union{Integer, NTuple{2,Integer}} = 1, bias::Bool = true, σ = identity, T::Type{<:Real} = Float32)
    k1, k2 = kernel # rozmiar jądra (height, width)
    cin, cout = pair # kanały: wejściowe i wyjściowe
    #powinien xavier powinien byćć!!!!!!!!!!!!!!
    w = randn(T, k1, k2, cin, cout) ./ sqrt(T(k1 * k2 * cin)) # inicjalizacja wag filtra; skala ~1/sqrt(fan_in), gdzie fan_in = k1 * k2 * cin
    b = bias ? zeros(T, cout, 1) : zeros(T, 0, 1) # gdy bias=false to trzyma pustą tablicę (upraszcza typ)
    _s = stride isa Integer ? (Int(stride), Int(stride)) : (Int(stride[1]), Int(stride[2])) # normalizacja kroku splotu do NTuple{2,Int}
    _p = pad isa Integer ? (Int(pad), Int(pad)) : (Int(pad[1]), Int(pad[2])) # normalizacja padding do NTuple{2,Int}
    return Conv{T, typeof(w), typeof(b), typeof(σ)}(w, b, σ, _s, _p, Workspace()) # jawna konstrukcja typu dla poprawnego wnioskowania typów przez kompilator; "Workspace()" = puste bufory robocze (P1)
end

# MaxPool - konstruktor
# Domyślnie "stride = pool" (niezachodzące okna), "pad = 0".
function MaxPool(pool::NTuple{N,Int}; stride::NTuple{N,Int} = pool,
                 pad::NTuple{N,Int} = ntuple(_ -> 0, N)) where {N}
    return MaxPool{N}(pool, stride, pad, Workspace()) # wrapper utrzymujący API spójne z Flux; "Workspace()" = bufory robocze (P1)
end

# Funkcja (c::Conv)(x::GraphNode) buduje w grafie operację konwolucji 2D:
# wagi "c.weight" (i opcjonalnie "c.bias") są opakowane w "Variable"
# (dzielą pamięć z warstwą - "optimize!" aktualizuje je in-place),
# meta-parametry ("pad", "stride") są przekazywane do operatora przez "Constant(c)".

# Wymiary:
# - "x.output" : "(H, W, C_in, B)" (wejście: wysokość, szerokość, kanały wejściowe, batch)
# - "c.weight" : "(kH, kW, C_in, C_out)" (wagi: wysokość, szerokość, kanały wejściowe, kanały wyjściowe)
# - wynik       : "(H_out, W_out, C_out, B)", gdzie "H_out = ⌊(H + 2P - kH)/S⌋ + 1" (wysokość wyjścia)
function (c::Conv)(x::GraphNode)
    W = Variable(c.weight; name="Conv.W") # trenowalne wagi (współdzielone z "c.weight")
    config = Constant(c) # meta-parametry ("pad", "stride", aktywacja, itd.) jako nietrenowalny liść
    y = if length(c.bias) > 0
        b = Variable(c.bias; name="Conv.b") # trenowalny bias gdy istnieje
        BroadcastedOperator(conv_op, config, x, W, b; name="conv+b") # wariant operatora z biasem
    else
        BroadcastedOperator(conv_op, config, x, W; name="conv") # wariant bez biasu
    end
    return c.σ === identity ? y : BroadcastedOperator(c.σ, y; name="σ") # aktywacja jako osobny węzeł (lub no-op gdy identity)
end


# Funkcja (m::MaxPool)(x::GraphNode) buduje w grafie operację pooling maksymalizujący 2D:
# Buduje węzeł "BroadcastedOperator(maxpool_op, cfg, x)", gdzie "cfg = Constant(m)" niesie "pool"/"stride"/"pad".
# Backward przepuszcza gradient tylko do pozycji argmax każdego okna.

function (m::MaxPool)(x::GraphNode)
    cfg = Constant(m)                                                      # konfiguracja poolingowa jako stała (nietrenowalna)
    return BroadcastedOperator(maxpool_op, cfg, x; name="maxpool")        # właściwy operator maxpool w grafie
end

# Funkcja (::FlattenLayer)(x::GraphNode) buduje w grafie operację spłaszczenia wszystkich wymiarów oprócz ostatniego (batcha):
# "(D1, D2, …, D_{n-1}, B) -> (D1·…·D_{n-1}, B)".
# Forward to "reshape" (bez kopii wartości), backward - "reshape" gradientu z powrotem do kształtu wejścia.
function (::FlattenLayer)(x::GraphNode)
    return BroadcastedOperator(flatten_op, x; name="flatten") # flatten realizowany jako operator AD (forward+backward w "autodiff.jl")
end

# Funkcja (d::Dropout)(x::GraphNode) buduje w grafie operację dropout:
# "BroadcastedOperator(dropout_op, Constant(d), x)" - węzeł operacji dropout.
# Maska losowana jest przy każdym wywołaniu "forward!" (w trybie treningu - "d.active == true").
# W trybie ewaluacji ("d.active == false") operator jest identycznością.
function (d::Dropout)(x::GraphNode)
    cfg = Constant(d) # referencja do warstwy Dropout (zawiera "active" i cache maski)
    return BroadcastedOperator(dropout_op, cfg, x; name = "dropout") # operator losujący maskę w forward i używający jej w backward
end

# trainmode!(model, on::Bool = true)
# testmode!(model)

# Przełączają tryb wszystkich "Dropout" w modelu (rekursywnie po "Chain").
# Dla warstw bez stanu treningu funkcja jest nie-operatorem.
trainmode!(d::Dropout, on::Bool = true) = (d.active = on; d) # modyfikacja stanu warstwy: training/eval
trainmode!(c::Chain, on::Bool = true) = (for l in c.layers; trainmode!(l, on); end; c) # propagacja trybu po wszystkich podwarstwach
trainmode!(x, on::Bool = true) = x # fallback: warstwa bez trybu treningowego pozostaje bez zmian (nie-operatorowa funkcja)
testmode!(model) = trainmode!(model, false) # alias semantyczny "przełącz na ewaluację"

# DataLoader - iteracja po batchach po ostatnim wymiarze
# Dla:
# - wektor 1D: ostatni wymiar jest indeksem elementu,
# - tensor n-D: próbki leżą na osi "ndims(X)" (ostatniej).
# To pasuje jednocześnie do:
# - danych tabelarycznych,
# - obrazów "(H, W, C, B)" dla CNN.

# Liczba próbek:
# - dla wektora: długość,
# - dla tablicy n-D: rozmiar ostatniej osi,
# - dla tuple (X, Y, ...): bierze pierwszą składową (zakłada spójność długości).
_nsamples(X::AbstractVector) = length(X) # 1D
_nsamples(X::AbstractArray)  = size(X, ndims(X)) # n-D
_nsamples(data::Tuple)       = _nsamples(first(data)) # para (X, Y)

# Wybieranie podbatcha po ostatnim wymiarze:
_select_last(X::AbstractVector, range) = X[range] # wektor: zwykłe indeksowanie zakresowe
function _select_last(X::AbstractArray, range)
    colons = ntuple(_ -> Colon(), ndims(X) - 1) # budujemy "(:, :, ..., :)" dla wszystkich osi poza ostatnią
    return X[colons..., range] # ostatnia oś dostaje "range"
end
_select_last(data::Tuple, range) = map(x -> _select_last(x, range), data) # ta sama selekcja do każdej składowej tuple (X, Y, ...)

# Funkcja DataLoader(data::Tuple; batchsize::Integer=1, shuffle::Bool=false) zwraca iterator po batchach "(X_batch, Y_batch)".
# Gdy "shuffle=true", losuje permutację "randperm(N)" i stosuje ją raz do obu składowych "data".
function DataLoader(data::Tuple; batchsize::Integer = 1, shuffle::Bool = false)
    if shuffle
        N = _nsamples(data) # liczba próbek
        idx = randperm(N) # jedna wspólna permutacja indeksów (ważne: X i Y muszą się przemieszczać razem)
        data = _select_last(data, idx) # wybranie potasowanych danych już w konstruktorze
    end
    return DataLoader{typeof(data)}(data, Int(batchsize), shuffle) # typ danych parametrycznie dla wydajności iteracji
end

Base.length(dl::DataLoader) = cld(_nsamples(dl.data), dl.batchsize) # liczba batchy = ceil(N / batchsize)

function Base.iterate(dl::DataLoader, state::Int = 1)
    N = _nsamples(dl.data) # całkowita liczba próbek
    state > N && return nothing # koniec iteracji
    last_idx = min(state + dl.batchsize - 1, N) # domknięcie ostatniego batcha (może być niepełny)
    return (_select_last(dl.data, state:last_idx), last_idx + 1) # (batch, następny stan)
end
