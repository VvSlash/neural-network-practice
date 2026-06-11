# src/structures.jl
# Typy definiujące graf obliczeniowy i warstwy sieci neuronowej.
# Pola "output"/"gradient" są typu "Union{Nothing, T}" – pozwala to startować od "nothing"
# i później (podczas "forward!"/"backward!") wpisać wartość typu T bez użycia "Any".

# Słownik pojęć (uczenie maszynowe) - pierwsze wystąpienia w tym pliku:
# - batch (paczka)       - grupa próbek przetwarzanych łącznie w jednym kroku; w tensorach zajmuje
#                          ostatni wymiar (oś batcha); "batchsize" = liczba próbek w paczce.
# - bias                 - wyraz wolny warstwy, dodawany do wyniku liniowego: y = W*x + b.
# - jądro / filtr        - mała macierz wag splotu (np. 3x3) przesuwana po obrazie; uczony detektor wzorca.
# - kanały (channels)    - liczba "warstw" wartości na piksel (1 = skala szarości; po splocie = liczba map cech).
# - stride (krok)        - co ile pozycji przesuwane jest okno splotu/poolingu.
# - padding (margines)   - wirtualne zera dokładane wokół wejścia w celu kontroli rozmiaru wyjścia.
# - pooling (MaxPool)    - redukcja rozdzielczości map cech: z każdego okna zostaje wartość maksymalna.
# - flatten              - spłaszczenie tensora do macierzy "(cechy, batch)" przed warstwą gęstą (Dense).
# - dropout              - regularyzacja: losowe zerowanie części aktywacji podczas treningu
#                          (maska = tablica true/false wskazująca zachowane elementy).
# - embedding (osadzenie)- mapowanie indeksów dyskretnych (np. słów) na gęste wektory liczb.

# Graph obliczeniowy

abstract type GraphNode end
abstract type Operator <: GraphNode end

# Dozwolone typy wartości krążących w grafie (wyjścia i gradienty).
const NodeValue = Union{Real, AbstractArray}

# Liść grafu o niezmiennej wartości (wejście/cel). 
# Pole "output" może być tablicą modyfikowaną w miejscu (".output .= new_value").
struct Constant{T} <: GraphNode
    output::T # wartość liścia (wejście/etykieta); zwykle "AbstractArray" lub "Real"
end

# Trenowalny liść grafu (waga, bias).
# Przed pierwszym "backward!" "gradient === nothing";
# Po nim trzyma wartość typu G (domyślnie T, bo "zero(output)"/"_seed_like" w "autodiff.jl/backward!" zachowują typ).
mutable struct Variable{T, G} <: GraphNode
    output::T # wartość parametru (waga/bias); aktualizowana in-place przez "optimize!"
    gradient::Union{Nothing, G} # gradient po "backward!"; "nothing" przed pierwszym backward lub po "zerograd!" dla operatorów
    name::String # etykieta diagnostyczna (np. "W" - waga, "b" - bias, "Conv.W" - waga konwolucji)
end
Variable(output::T; name::AbstractString="?") where {T} = Variable{T, T}(output, nothing, String(name)) # Zmienna o wartości "output" i typie "T", gradientie "nothing" i nazwie "name".

# Operator o dokładnie dwóch wejściach skalarnych (np. strata, iloczyn skalarny).
# Domyślnie T = G = Real
# Ogólne ograniczenie "Real" wystarcza, bo konkretny typ wyjścia zależy od wejść dopiero przy forward.
mutable struct ScalarOperator{F, T<:Real, G<:Real} <: Operator
    inputs::NTuple{2, GraphNode} # dokładnie dwa węzły-wejścia (dowolnego podtypu "GraphNode")
    output::Union{Nothing, T} # skalarny wynik "forward"; "nothing" dopóki nie zawołano "forward!"
    gradient::Union{Nothing, G} # skalarny gradient po "backward!"; "nothing" na starcie i po "zerograd!"
    gradbuf::Union{Nothing, G} # zachowany bufor gradientu (P1/B4); reużywany między przejściami backward!
    name::String # etykieta diagnostyczna (np. "sop" - scalar operator)
end
ScalarOperator(fun::F, inputs::NTuple{2, GraphNode}; name::AbstractString="?") where {F} = ScalarOperator{F, Real, Real}(inputs, nothing, nothing, nothing, String(name)) # Operator o funkcji "fun" i dwóch wejściach "inputs" i nazwie "name".

# Operator z dowolną liczbą wejść (operacje element-wise, "*", "+", redukcje):
# 1 wejście - BroadcastedOperator(relu, x), BroadcastedOperator(sum, x), BroadcastedOperator(flatten_op, x)
# 2 wejścia - BroadcastedOperator(+, mul, b), BroadcastedOperator(*, W, x), BroadcastedOperator(bce_el, yhat, y), BroadcastedOperator(maxpool_op, cfg, x)
# 3 wejścia - BroadcastedOperator(conv_op, config, x, W)
# 4 wejścia - BroadcastedOperator(conv_op, config, x, W, b)
# Wynik może być tablicą albo skalarem  stąd "T, G <: NodeValue".
mutable struct BroadcastedOperator{F, T<:NodeValue, G<:NodeValue} <: Operator
    inputs::Tuple{Vararg{GraphNode}} # dowolna liczba węzłów-wejść (krotność wymuszona dopiero przez metody "forward"/"backward")
    output::Union{Nothing, T} # wynik "forward" (tablica lub skalar); "nothing" przed "forward!"
    gradient::Union{Nothing, G} # zakumulowany gradient po "backward!"; "nothing" na starcie i po "zerograd!"
    gradbuf::Union{Nothing, G} # zachowany bufor gradientu (P1/B4); własność wyłączna węzła, reużywany między przejściami backward! (gradient nigdy nie współdzieli pamięci z innym węzłem)
    name::String # etykieta diagnostyczna (np. "W*x+b" - mnożenie macierzowe, "conv+b" - konwolucja z biasem, "bce" - binary cross-entropy)
end
BroadcastedOperator(fun::F, inputs::Vararg{GraphNode}; name::AbstractString="?") where {F} = BroadcastedOperator{F, NodeValue, NodeValue}(inputs, nothing, nothing, nothing, String(name)) # Operator o funkcji "fun" i dowolnej liczbie wejściach "inputs" i nazwie "name".

# Przestrzeń robocza warstwy (optymalizacja P1):
# bufory wielokrotnego użytku między iteracjami treningu (np. macierze im2col, gradienty wejść).
# Mapa "rola -> tablica"; bufor jest pozyskiwany przez "_fit!" w "autodiff.jl"
# i realokowany tylko przy zmianie kształtu/typu (np. inny batchsize w ewaluacji).
struct Workspace
    bufs::Dict{Symbol, Array} # bufor per rola (":cols", ":gx", ...); konkretny typ przywracany asercją przy odczycie
end
Workspace() = Workspace(Dict{Symbol, Array}())

# Warstwy i Chain

# Sekwencja warstw. "Chain(l1, l2, ...)" lub "Chain((l1, l2, ...))"
struct Chain{L<:Tuple}
    layers::L # niejednorodna krotka warstw wywoływanych kolejno: "layer_n(…layer_1(x)…)" (np. Chain(Dense(32 => 16, relu), Dense(16 => 8, relu)) lub Chain((Dense(32 => 16, relu), Dense(16 => 8, relu))))
end
Chain(layers...) = Chain{typeof(layers)}(layers)

# Warstwa w pełni połączona: "y = sigma.(W*x .+ b)"
struct Dense{T<:Real, A<:AbstractMatrix{T}, B<:AbstractVecOrMat{T}, F}
    weight::A # macierz wag "output x input" (trenowalna, opakowana w "Variable" przy budowie grafu)
    bias::B # bias "output x 1" (trenowalny); konwencja kolumnowa: "x: features x batch"
    σ::F # funkcja aktywacji stosowana na końcu (np. "relu", "sigmoid", "identity")
end

# Warstwa osadzeń (embedding): wybiera kolumny z macierzy "weight"
struct Embedding{T<:Real, M<:AbstractMatrix{T}}
    weight::M # macierz osadzeń "dimension x vocabulary"; wybór kolumn realizuje "lookup" po indeksach
end

# Warstwa konwolucyjna 2D.
struct Conv{T<:Real, W<:AbstractArray{T, 4}, B<:AbstractVecOrMat{T}, F}
    weight::W # 4-D jądro "(kernel_height, kernel_width, channels_in, channels_out)" (trenowalne)
    bias::B # bias na kanał "channels_out x 1"; gdy "bias=false" w konstruktorze – ma zerową długość
    σ::F # aktywacja stosowana po konwolucji (np. "relu", "identity")
    stride::NTuple{2, Int} # krok splotu "(sH, sW)" – osobno dla wysokości i szerokości
    pad::NTuple{2, Int} # padding (symetryczny) "(pH, pW)" – osobno dla wysokości i szerokości
    ws::Workspace # bufory robocze splotu (im2col, GEMM, gradienty) współdzielone między iteracjami
end

# Pooling maksymalizujący.
struct MaxPool{N}
    pool::NTuple{N, Int} # rozmiar okna w każdym z "N" wymiarów przestrzennych (np. "(height, width)")
    stride::NTuple{N, Int} # krok okna per-wymiar; domyślnie równy "pool"
    pad::NTuple{N, Int} # padding per-wymiar; poza brzegiem traktowany jak "-Inf"
    ws::Workspace # bufory robocze: ":gx" (gradient wejścia), ":ihi"/":iwi" (cache argmax z forward)
end

# Spłaszczenie (height, width, channels, batch) -> (height*width*channels, batch).
struct FlattenLayer end
const flatten = FlattenLayer()

# Warstwa Dropout z odwróconym skalowaniem.
# Maska jest cache'owana między "forward!" i "backward!" tego samego przebiegu.
mutable struct Dropout
    p::Float32 # prawdopodobieństwo wyzerowania pojedynczego elementu (0..1)
    active::Bool # "true" = trening (losowanie + skalowanie), "false" = ewaluacja (identyczność)
    mask::Union{Nothing, BitArray} # ostatnio wylosowana maska "rand(T, size(x)) .> p"; "nothing" przed forward lub w trybie eval
    ws::Workspace # bufor roboczy losowania (":rand") współdzielony między iteracjami
end
Dropout(p::Real) = Dropout(Float32(p), true, nothing, Workspace())

# DataLoader

# Iterator po batchach z ostatniego wymiaru danych.
# Dla "data::Tuple{X, Y}" iteruje parami: "X" cięte po ostatnim wymiarze
# (dla tensorów "(height, width, channels, batch)" bierze po "batch"), "Y" analogicznie.
# Obsługa "shuffle" - implementacja w "layers.jl".

struct DataLoader{D}
    data::D # zwykle "Tuple{X, Y}"; iteracja po ostatnim wymiarze każdej składowej
    batchsize::Int # liczba próbek w jednym batchu
    shuffle::Bool # flaga pamiętająca, czy dane zostały przepermutowane w konstruktorze
end
