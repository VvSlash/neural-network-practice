# src/autodiff.jl
# Reverse-mode AD (autograd) na statycznym grafie obliczeniowym.

using LinearAlgebra: mul! # mnożenie macierzowe in-place (BLAS) dla splotu im2col+GEMM
using Random: rand! # losowanie w miejscu do istniejącego bufora (Dropout)

# Pomocnicze zarządzanie buforami (optymalizacja P1):

# Zwraca "out", jeśli pasuje typem i kształtem; w przeciwnym razie świeżą tablicę.
# Pierwsze wywołanie jądra (out === nothing) zawsze alokuje; kolejne reużywają bufor.
_ensure(out, ::Type{T}, dims::Dims{N}) where {T, N} =
    (out isa Array{T, N} && size(out) == dims) ? out : Array{T, N}(undef, dims)

# Pozyskuje z przestrzeni roboczej bufor o roli "key", typie "T" i kształcie "dims".
# Realokacja tylko przy zmianie typu/kształtu (np. inny batchsize przy ewaluacji).
function _fit!(ws::Workspace, key::Symbol, ::Type{T}, dims::Dims{N}) where {T, N}
    buf = get(ws.bufs, key, nothing)
    buf isa Array{T, N} && size(buf) == dims && return buf::Array{T, N} # trafienie - bufor pasuje
    fresh = Array{T, N}(undef, dims) # nowy bufor (pierwsze użycie lub zmiana kształtu)
    ws.bufs[key] = fresh
    return fresh
end

# Idea: wyrażenie (np. strata L) jest reprezentowane jako skierowany graf acykliczny (ang. directed acyclic graph, DAG) węzłów.
#   - liście: "Constant" (dane wejściowe, etykiety, konfiguracja warstw) i "Variable" (trenowalne parametry: wagi, biasy),
#   - wewnętrzne węzły: "Operator" (ScalarOperator / BroadcastedOperator)
#     przechowujące funkcję "fun" i wejścia.
# Forward: wylicza "output" każdego operatora w porządku topologicznym (od liści do korzenia).    
# Backward: propaguje gradient w odwrotnej kolejności (od korzenia do liści),
# akumulując wkłady do ".gradient" każdego węzła trenowalnego (Variable) zgodnie z regułą łańcuchową.

# Budowa grafu w porządku topologicznym:

# "graph(root)" - dla korzenia (np. węzła straty) zwraca listę wszystkich jego przodków w kolejności topologicznej: 
# liście pojawią się pierwsze, korzeń ostatni. Dzięki temu "forward!" może po prostu iterować po liście  w kolejności, a "backward!" w odwrotnej.
function graph(root::GraphNode)
    ordered = GraphNode[] # tu budowany jest wynik (kolejność topologiczna)
    seen = IdDict{GraphNode, Bool}() # słownik "węzeł -> czy odwiedzony"; IdDict używa identyczności (===), nie "=="
    _visit!(root, seen, ordered) # DFS (Depth-First Search) od korzenia w dół; dopisuje (w miejscu) do słownika "seen" wartość "true" i listy "ordered" węzły (odwiedzane węzły są dodawane do "ordered")
    return ordered # zwraca gotową listę w porządku topologicznym
end

# DFS z pamięcią podręczną:
# Każdy węzeł jest odwiedzany dokładnie raz, nawet gdy w grafie skierowanym acyklicznym (DAG) występuje więcej niż jedna ścieżka do niego
function _visit!(n::GraphNode, seen::IdDict{GraphNode, Bool}, ordered::Vector{GraphNode})
    haskey(seen, n) && return nothing # jeśli już odwiedziliśmy ten węzeł - stop (to znaczy, że jest cykl w grafie)
    seen[n] = true # oznacza jako odwiedzony ZANIM zejdzie w dół (inaczej rekurencja mogłaby się zapętlić)
    if n isa Operator # tylko operatory mają wejścia - liście ("Constant"/"Variable") to koniec rekurencji
        for inp in n.inputs # zchodzi rekurencyjnie do każdego wejścia operatora
            _visit!(inp, seen, ordered) # wejście dodane do "ordered" PRZED bieżącym węzłem
        end
    end
    push!(ordered, n) # po odwiedzeniu wszystkich dzieci dodajemy siebie - to daje kolejność topologiczną (bo zawsze na końcu)
    return nothing
end

# Przejście w przód:

# Przechodzi w porządku topologicznym od początku (liście -> korzeń) i wypełnia pole "output" każdego operatora, korzystając z już wyliczonych wartości wejść.
# Liście (Constant/Variable) już mają gotowe wartości w polu "output" - ich "_compute!" to no-op (nic nie liczy).

function forward!(order::Vector{GraphNode})
    for n in order # iteracja w kolejności topologicznej: najpierw liście, na końcu korzeń
        _compute!(n) # Julia wybiera metodę po typie węzła: liście (Constant/Variable) nic nie robią; dla operatorów wylicza wynik i zapisuje go w n.output przez forward(...)
    end
    return last(order).output # wartość korzenia = wynik całego wyrażenia (np. wartość straty L (loss value)) (zawsze na końcu - w korzeniu)
end

# Liście nic nie liczą - ich "output" jest zdefiniowane jako "input.output .= x" w pętli treningowej, albo wagi z konstruktora Dense).
_compute!(::Constant) = nothing
_compute!(::Variable) = nothing

# Operator z dowolną liczbą wejść: zbiera wartości wszystkich wejść i
# wywołuje metodę "forward(n, args...)", która jest wybierana po
# "typeof(fun)" (np. "+", "*", "relu", "σ", "conv_op", itp. itd.).
# Optymalizacja P1: pierwsze wywołanie alokuje wyjście przez "forward",
# kolejne piszą w miejscu do istniejącego bufora "n.output" przez "forward!";
# przy zmianie kształtu jądra in-place same realokują bufor ("_ensure").
function _compute!(n::BroadcastedOperator)
    # "ntuple(f, N)" tworzy krotkę "(f(1), ..., f(N))"; tu zbiera ".output" z każdego wejścia.
    # Rozpakowanie "..." rozwija krotkę na argumenty pozycyjne "forward"/"forward!".
    vals = ntuple(i -> n.inputs[i].output, length(n.inputs))
    out = n.output
    n.output = out === nothing ? forward(n, vals...) : forward!(out, n, vals...)
    return nothing
end

# Wariant domyślny "forward!": operator bez jądra in-place liczy jak dotychczas (alokując).
# Jądra in-place dla gorących operatorów ("+", "*", "relu", "σ", conv, maxpool, dropout) niżej.
forward!(out, n::BroadcastedOperator, vals...) = forward(n, vals...)

# Operator skalarny ma dokładnie dwa wejścia - bez ntuple, prostsza wersja.
function _compute!(n::ScalarOperator)
    n.output = forward(n, n.inputs[1].output, n.inputs[2].output) # zbiera wartości wejść i wywołuje metodę "forward(n, args...)" zgodnie z typem operatora
    return nothing
end

# Przejście w tył:
# 1. Zerowanie ".gradient" we wszystkich węzłach (Variable dostaje zera
# tego samego kształtu co output, Operator dostaje "nothing" - jako placeholder),
# 2. Na korzeniu ustawia gradient początkowy ("seed" - zwykle 1 przy skalarnej stracie, bo dL/dL = 1),
# 3. Przechodzi w odwrotnej kolejności topologicznej: dla każdego operatora
# liczy lokalny gradient po wejściach (reguła łańcuchowa) i akumulujemy go w polu ".gradient" każdego wejścia.

# 1. Zerowanie gradientów:

# "zerograd!" wywołuje pomocniczą metodę "_zerograd!" z wybieraniem metody po typie węzła.
# Każda gałąź jest oddzielną metodą i Julia kompiluje ją bezpośrednio.
function zerograd!(order::Vector{GraphNode})
    for n in order # iteracja po wszystkich węzłach grafu
        _zerograd!(n) # wybieranie metody po typie węzła: Constant -> no-op, Variable -> zeros, Operator -> nothing
    end
    return nothing
end

_zerograd!(::Constant) = nothing # Constant jest liściem nie-trenowalnym - nie ma gradientu
# Dla parametrów: reużycie istniejącego bufora gradientu (fill! zamiast alokacji "zero(output)") - optymalizacja P1 (B5).
# Bufor jest własnością wyłączną Variable (tworzony tu raz), więc zerowanie w miejscu jest bezpieczne.
function _zerograd!(n::Variable)
    g = n.gradient
    if g isa AbstractArray
        fill!(g, zero(eltype(g))) # zerowanie w miejscu - bez alokacji
    else
        n.gradient = _zero_like(n.output) # pierwsza iteracja (gradient === nothing) lub parametr skalarny
    end
    return nothing
end
_zerograd!(n::Operator) = (n.gradient = nothing; nothing) # dla operatorów: "nothing" = "gradient jeszcze nie przyszedł" (sygnał dla "_accumulate!")

# Tworzenie "neutralnego" zera w typie i kształcie danej wartości.
_zero_like(x::AbstractArray) = zero(x) # "zero(A)" dla tablicy zwraca tablicę tego samego kształtu/typu wypełnioną zerami
_zero_like(x::Real) = zero(x) # "zero(::T) where T<:Real" zwraca "T(0)"

# 2. + 3. Właściwy backward:

# "backward!" rozpoczyna propagację od korzenia z gradientem "seed" i schodzi do liści w odwrotnej kolejności topologicznej.
function backward!(order::Vector{GraphNode}; seed=1)
    zerograd!(order) # zeruje wszystkie ".gradient"
    root = last(order) # korzeń w porządku topologicznym = ostatni element listy = węzeł straty L
    root.gradient = _seed_like(root.output, seed) # dL/dL = seed, kształt/typ dopasowany do "root.output"

    for n in Iterators.reverse(order) # propagacja gradientu w odwrotnej kolejności topologicznej (od korzenia do liści)
        n isa Operator || continue # Pominięcie liści, które nie produkują gradientu (nie są operatorami) dla innych węzłów (nie mają wejść)
        n.gradient === nothing && continue # Pominięcie operatorów, które jeszcze nie otrzymały gradientu (gałąź odcięta od "root")
        in_values = ntuple(i -> n.inputs[i].output, length(n.inputs)) # wartości wejść z forward (potrzebne do liczenia gradientu lokalnego)
        in_grads = backward(n, in_values..., n.gradient) # reguła łańcuchowa: wybieranie metody po typie operatora "backward(n, ...)"
        # zwraca krotkę "(g_in1, g_in2, ...)" - po jednym elemencie na każde wejście
        for (inp, g) in zip(n.inputs, in_grads) # gradienty do wejść operatora
            _accumulate!(inp, g) # "+=" z obsługą "nothing" (np. stałych wejść, które nie uczestniczą w automatycznym różniczkowaniu)
        end
    end
    return nothing
end

# Inicjalizacja gradientu korzenia przez dopasowanie kształtu:
_seed_like(x::AbstractArray, s) = fill!(similar(x), s) # tablica tego samego kształtu co "x", wypełniona "s" ("seed" - zwykle 1)
_seed_like(x::Real, s) = convert(typeof(x), s) # dla skalarnej straty: po prostu "s" w typie "x" ("seed" - zwykle 1)

# Akumulacja gradientu w węzłach:

# Cztery metody pokrywają wszystkie sensowne kombinacje (warianty):
# - Constant: nic nie akumuluje (Constant nie ma pola gradient),
# - (Variable|Operator) + Nothing: też nic (brak gradientu do dodania),
# - (Variable|Operator) + NodeValue: zapisanie nowego gradientu lub dodanie do istniejącego.

_accumulate!(::Constant, _) = nothing # Constant nigdy nie dostaje gradientu
_accumulate!(::Variable, ::Nothing) = nothing   # "nothing" = brak gradientu do dodania (pominięcie)
_accumulate!(::Operator, ::Nothing) = nothing   # "nothing" = brak gradientu do dodania (pominięcie)

# Akumulacja dla Variable (parametrów trenowalnych) - w miejscu (optymalizacja P1).
# Bezpieczeństwo aliasingu: bufor "n.gradient" jest własnością wyłączną Variable
# (tworzony w "zerograd!"), a "g" pochodzi ze świeżych lub roboczych tablic jąder
# backward (np. "g*Bᵀ", bufor ":gW" splotu) - nigdy nie jest tym samym buforem.
function _accumulate!(n::Variable, g::NodeValue)
    buf = n.gradient
    if buf isa AbstractArray && g isa AbstractArray && size(buf) == size(g)
        buf .+= g # sumowanie w miejscu do własnego bufora - bez alokacji
    elseif buf === nothing
        n.gradient = g # przypadek brzegowy: pierwszy gradient przed "zerograd!"
    else
        n.gradient = buf .+ g # przypadek brzegowy: parametr skalarny lub niezgodny kształt
    end
    return nothing
end

# Analogiczna akumulacja dla operatorów. Różni się od Variable tylko typem pola ".gradient"
# (ten sam wzorzec logiki - ale typ pola ".gradient" jest inny: NodeValue).
function _accumulate!(n::Operator, g::NodeValue)
    if n.gradient === nothing # nadpisanie pierwszego przychodzącego gradient
        n.gradient = g
    else
        n.gradient = n.gradient .+ g  # sumowanie kolejnych gradientów po wyjściach operatora (wiele wyjść operatora = suma gradientów po nich)
    end
    return nothing
end

# Optymalizator (SGD w miejscu):
# Stochastic gradient descent: theta+1 = theta - eta * dL/dtheta (L - strata, eta - learning rate w czasie t, theta - parametr trenowalny).
# Dla każdej Variable trenowalnej modyfikuje jej "output" w miejscu (".-=").
# Tablica jest dzielona z warstwą (np. "Variable(dense.weight)" w "(d::Dense)(x)" - wskazuje na te same komórki pamięci; update jednego = update drugiego).
function optimize!(order::Vector{GraphNode}, eta)
    for n in order
        # Pominięcie Constant (nietrenowalne), operatorów (nie mają parametrów) i Variable bez gradientu (np. w grafie odciętym od straty (dany węzeł/parametr nie ma ścieżki do węzła straty L)).
        (n isa Variable && n.gradient !== nothing) || continue
        if n.output isa AbstractArray
            # Tablica: konwertuje krok "eta" do typu elementu parametru
            # (np. eta::Float64 -> Float32 dla wag), żeby uniknąć promocji typu przy "-=".
            eta_typed = convert(eltype(n.output), eta)
            n.output .-= eta_typed .* n.gradient   # "theta -= eta * g" dla każdego elementu tablicy w miejscu
        else
            # Skalarny parametr: analogicznie, ale bez rzutowania typu.
            # "-=" dla niemutowalnego skalara przypisuje pole (wymaga "mutable struct").
            n.output -= convert(typeof(n.output), eta) * n.gradient
        end
    end
    return nothing
end

# Aktywacje - skalarne funkcje pomocnicze
# Potrzebne do "typeof(relu)", "typeof(σ)" i wybierania po nich odpowiednich metod "forward" / "backward".

relu(x::Real) = max(zero(x), x) # ReLU(x) = max(0, x); "zero(x)" zamiast "0" daje zgodność typu (Float32/64 itd.)
σ(x::Real) = one(x) / (one(x) + exp(-x)) # sigmoid(x) = 1/(1+e^-x); "one(x)" - zgodność typu (Float32/64 itd.)

# Forward / Backward dla konkretnych operatorów
# Dla każdej operacji grafu są dwie metody:
#   forward(::BroadcastedOperator{typeof(fun)}, inputs...) -> wartość wyjścia,
#   backward(::BroadcastedOperator{typeof(fun)}, inputs..., g) -> krotka gradientów po wejściach.
# "g" to gradient przychodzący od węzłów "wyżej" (reguła łańcuchowa).
# Gradienty są zwracane po KOLEJNOŚCI wejść (zip z "n.inputs" w backward!).

# Dodawanie każdego elementu (z rzutowaniem biasu)
forward(::BroadcastedOperator{typeof(+)}, a, b) = a .+ b # rzutowanie rozwiązuje różne kształty (np. macierz + wektor-bias)
function backward(::BroadcastedOperator{typeof(+)}, a, b, g)
    # Dla "y = a + b": dy/da = 1, dy/db = 1 -> gradient przechodzi bez dodatkowego przeskalowania/zmiany wartości lokalnie przez samą operację +.
    # Jeśli "a" lub "b" mają mniejszy kształt (bias ma wymiar 1 po batchu), to gradient "g" trzeba zredukować (zsumować) po wymiarach, gdzie było rzutowanie.
    return (_reduce_to(g, a), _reduce_to(g, b))
end

# Dodawanie w miejscu: wynik rozgłoszenia pisany do bufora "out" (P1).
function forward!(out, ::BroadcastedOperator{typeof(+)}, a::AbstractArray, b::AbstractArray)
    dims = map(length, Broadcast.combine_axes(a, b)) # kształt wyniku rozgłoszenia (np. macierz + bias)
    o = _ensure(out, promote_type(eltype(a), eltype(b)), dims) # bufor wyjścia (realokacja tylko przy zmianie kształtu)
    o .= a .+ b # zapis w miejscu - bez alokacji
    return o
end

# Mnożenie macierzowe: y = A * B
forward(::BroadcastedOperator{typeof(*)}, A, B) = A * B # samo "*" to mnożenie macierzy, a nie każdego elementu
backward(::BroadcastedOperator{typeof(*)}, A, B, g) = (g * B', A' * g) # dL/dA = g*Bᵀ,  dL/dB = Aᵀ*g

# Mnożenie macierzowe w miejscu: "mul!" (BLAS) pisze wprost do bufora "out" (P1).
function forward!(out, ::BroadcastedOperator{typeof(*)}, A::AbstractMatrix, B::AbstractMatrix)
    o = _ensure(out, promote_type(eltype(A), eltype(B)), (size(A, 1), size(B, 2)))
    mul!(o, A, B) # BLAS bez tablicy pośredniej
    return o
end

# ReLU (każdego elementu)
forward(::BroadcastedOperator{typeof(relu)}, x) = max.(zero(eltype(x)), x) # "max.(0, x)" z zachowaniem typu elementu
backward(::BroadcastedOperator{typeof(relu)}, x, g) = (g .* (x .> 0),) # dReLU/dx = 1 jeśli x>0 inaczej 0; "x .> 0" to BitArray - rzutowanie działa

# ReLU w miejscu (P1).
function forward!(out, ::BroadcastedOperator{typeof(relu)}, x::AbstractArray)
    o = _ensure(out, eltype(x), size(x))
    o .= max.(zero(eltype(x)), x)
    return o
end

# Sigmoid: σ(x) = 1/(1+e^-x)
forward(::BroadcastedOperator{typeof(σ)}, x) = one(eltype(x)) ./ (one(eltype(x)) .+ exp.(-x))

# Sigmoid w miejscu (P1).
function forward!(out, ::BroadcastedOperator{typeof(σ)}, x::AbstractArray)
    o = _ensure(out, eltype(x), size(x))
    o .= one(eltype(x)) ./ (one(eltype(x)) .+ exp.(-x))
    return o
end
function backward(::BroadcastedOperator{typeof(σ)}, x, g)
    s = one(eltype(x)) ./ (one(eltype(x)) .+ exp.(-x)) # przeliczenie sigmoid(x) bo "forward" nie przekazuje wyniku
    return (g .* s .* (one(eltype(s)) .- s),) # dσ/dx = σ(x)*(1 − σ(x))
end

# Identyczność (neutralna aktywacja)
forward(::BroadcastedOperator{typeof(identity)}, x) = x # żadnej transformacji
backward(::BroadcastedOperator{typeof(identity)}, x, g) = (g,) # dx/dx = 1 -> gradient wprost

# Redukcja sumy do skalara: y = Σ x
forward(::BroadcastedOperator{typeof(sum)}, x) = sum(x) # zwraca skalar (węzeł "skalarny" operatora broadcastowego)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = (fill!(similar(x), g),) # dΣx/dx_i = 1 -> gradient rozprowadzany wszędzie; "similar" alokuje kształt "x"

# Binary cross-entropy (każdego elementu)
bce_el(yhat, y) = -(y * log(yhat) + (1 - y) * log(1 - yhat)) # skalarna forma; używana do wyboru przez Julię po "typeof(bce_el)"
forward(::BroadcastedOperator{typeof(bce_el)}, yhat, y) = -(y .* log.(yhat) .+ (1 .- y) .* log.(1 .- yhat)) # wersja po każdym elemencie przez rzutowanie
function backward(::BroadcastedOperator{typeof(bce_el)}, yhat, y, g)
    # Pochodna BCE po prognozie "yhat": dL/dyhat = (yhat − y)/(yhat(1−yhat)), zapisana w formie rozbitej "-y/yhat + (1−y)/(1−yhat)".
    gyhat = g .* (-(y ./ yhat) .+ (1 .- y) ./ (1 .- yhat))
    return (gyhat, nothing) # etykieta "y" nie jest trenowalna - gradient = "nothing"
end

# Numerycznie stabilny logit cross-entropy (softmax + cross-entropy jednocześnie)
# Wejścia: 
# "yhat" - surowe logity "C x B" (wyniki modelu dla klasy - dowolna liczba rzeczywista), 
# "y" - one-hot "C x B" (wektor etykiet).
# Wynik - skalar.
# Softmax(yhat)_i = exp(yhat_i) / Σ_j exp(yhat_j); CE = − Σ y_i * log softmax(yhat)_i.
function logitcrossentropy(yhat::AbstractMatrix, y::AbstractMatrix)
    maxv = maximum(yhat; dims=1) # max w każdej kolumnie (per próbka w batchu)
    shifted = yhat .- maxv # "zerowanie" maxa - zapobiega overflow "exp"
    logZ = log.(sum(exp.(shifted); dims=1)) # logarytm normalizatora softmaxu per kolumna
    logp = shifted .- logZ # log softmax(yhat) = shifted − logZ (tożsamość)
    return -sum(y .* logp) / size(yhat, 2) # CE uśrednione po batchu (dzielenie przez B = size(yhat, 2))
end

forward(::BroadcastedOperator{typeof(logitcrossentropy)}, yhat, y) = logitcrossentropy(yhat, y)  # wrapper dla grafu AD
function backward(::BroadcastedOperator{typeof(logitcrossentropy)}, yhat, y, g)
    # Gradient po logicie: dL/dyhat = (softmax(yhat) − y) / B; wystarczy softmax i suma.
    maxv = maximum(yhat; dims=1) # jak w forward - stabilność
    shifted = yhat .- maxv
    p = exp.(shifted) ./ sum(exp.(shifted); dims=1) # softmax w stabilnej formie
    B = size(yhat, 2) # rozmiar batcha (do uśrednienia)
    gyhat = (p .- y) .* (g / B) # (p − y)/B skalowane przez gradient nadchodzący z góry
    return (gyhat, nothing) # "y" nie jest trenowalne - gradient = "nothing"
end

# Pomocnicza redukcja gradientu do zadanego kształtu
# Przypadek użycia:
# dodawanie "y = W*x + b". "W*x" ma kształt "(out, batch)", "b" - "(out, 1)".
# Rzutowanie działa, ale gradient od "+" przychodzi jako "(out, batch)", a "b" oczekuje "(out, 1)".
# Trzeba zsumować po osiach, gdzie "b" był rozciągany (tu: po osi batcha).
function _reduce_to(g::AbstractArray, target::AbstractArray)
    size(g) == size(target) && return g # kształty się zgadzają
    tsize = ntuple(i -> i <= ndims(target) ? size(target, i) : 1, ndims(g)) # kształt target "wyrównany" do ndims(g) (dokłada 1 na brakujących osiach)
    dims = Tuple(i for i in 1:ndims(g) if tsize[i] == 1 && size(g, i) != 1) # osie, gdzie target miał wymiar 1 i g ma więcej -> trzeba zsumować
    reduced = isempty(dims) ? copy(g) : dropdims(sum(g; dims=dims); dims=dims)  # "sum(...; dims)" zachowuje osie 1 - "dropdims" je usuwa
    return reshape(reduced, size(target)) # upewnij się, że zwracamy DOKŁADNIE size(target)
end
_reduce_to(g, ::Real) = sum(g) # skalarny "target" (suma wszystkich elementów)

# Konwolucja 2D
# Wymiary:
#   wejście   x : (H, W, C_in, B)      - batch obrazów "Heigth x Width" z "Channels in" (kanały wejściowe)
#   jądro     W : (kH, kW, C_in, C_out) - "Channels out" filtrów o rozmiarze "Kernel Height x Kernel Width" (kanały wyjściowe)
#   wyjście   y : (H_out, W_out, C_out, B) - "Height out x Width out x Channels out x Batch" gdzie H_out = floor((H + 2*pH − kH)/sH) + 1  (analogicznie dla W). "pH/pW" - padding symetryczny, "sH/sW" - krok splotu per wymiar.
# Konwencja matematyczna (z flipem jądra): indeksuje "W" przez "(kH+1-kh, kW+1-kw, ...)".

# Funkcja-znacznik żeby "typeof(conv_op)" miał unikalną wartość używaną do wyboru metody w "forward" / "backward".
# W praktyce "Conv" zwraca "BroadcastedOperator(conv_op, …)".
function conv_op end

# Implementacja referencyjna ("naive"): splot na 7 zagnieżdżonych pętlach.
# Zachowana po optymalizacji P0 (im2col+GEMM poniżej) do testów poprawności
# i benchmarków "przed/po" w "scripts/benchmark_bottlenecks.jl".
function _conv_forward_naive(c, x::AbstractArray{Tx,4}, W::AbstractArray{Tw,4}) where {Tx, Tw}
    pH, pW = c.pad # margines per wymiar
    sH, sW = c.stride # krok per wymiar
    kH, kW, Cin, Cout = size(W) # rozmiary jądra i liczby kanałów (przesuwne okno / uczony detektor wzorca / filtr (np. 3 x 3))
    H, Wd, Cin2, B = size(x) # rozmiary przestrzenne, kanały, batch
    Cin == Cin2 || throw(DimensionMismatch("Conv: Cin z wag != Cin z wejścia")) # sprawdzenie czy liczba kanałów wejściowych jest zgodna z liczbą kanałów wejściowych
    H_out = div(H  + 2pH - kH, sH) + 1 # rozmiar wyjścia w osi Height
    W_out = div(Wd + 2pW - kW, sW) + 1 # rozmiar wyjścia w osi Width
    y = zeros(promote_type(Tx, Tw), H_out, W_out, Cout, B) # bufor wyniku; wspólny typ "Tx, Tw" (np. Float32 (x) Float32 = Float32)
    @inbounds for b = 1:B, cout = 1:Cout, h = 1:H_out, w = 1:W_out # pętla po wszystkich wyjściach (Height, Width, Channels out, Batch)
        s = zero(eltype(y)) # akumulator dla bieżącej komórki y[h, w, cout, b]
        for cin = 1:Cin, kh = 1:kH, kw = 1:kW # suma po kanałach wejścia i oknie jądra
            hi = (h - 1) * sH + kh - pH # indeks w "x" dla bieżącego elementu okna (z uwzględnieniem kroku i marginesu)
            wi = (w - 1) * sW + kw - pW # indeks w "W" dla bieżącego elementu okna (z uwzględnieniem kroku i marginesu)
            if 1 <= hi <= H && 1 <= wi <= Wd # elementy poza "x" nie wnoszą wkładu (zera jako margines)
                s += W[kH + 1 - kh, kW + 1 - kw, cin, cout] * x[hi, wi, cin, b]  # splot = suma iloczynów i flip indeksu "W"
            end
        end
        y[h, w, cout, b] = s # zapis zsumowanej wartości
    end
    return y
end

# Backward referencyjny ("naive"): jednoczesne dL/dx ("gx") i dL/dW ("gW")
# ta sama pętla z gradientem "g" przychodzącym od wyjścia (z góry).
# Zachowany po optymalizacji P0 - rola jak przy "_conv_forward_naive".
function _conv_backward_naive(c, x::AbstractArray{Tx,4}, W::AbstractArray{Tw,4}, g::AbstractArray{Tg,4}) where {Tx, Tw, Tg}
    pH, pW = c.pad # margines
    sH, sW = c.stride # krok
    kH, kW, Cin, Cout = size(W) # rozmiary jądra
    H, Wd, _, B = size(x) # rozmiary wejścia (kanały "Cin" są w W)
    H_out, W_out, _, _ = size(g) # rozmiary gradientu wyjścia (odpowiada kształtowi y z forwarda)
    gx = zeros(Tx, size(x)) # bufor na gradient po wejściu; typ jak "x"
    gW = zeros(Tw, size(W)) # bufor na gradient po wagach; typ jak "W"
    @inbounds for b = 1:B, cout = 1:Cout, h = 1:H_out, w = 1:W_out # pętla po wszystkich wyjściach (Height, Width, Channels out, Batch)
        go = g[h, w, cout, b] # gradient wyjścia (output) "z góry" dla komórki y[h, w, cout, b]
        for cin = 1:Cin, kh = 1:kH, kw = 1:kW # pętla po każdej parze (wejście, element jądra) która wnosiła wkład do y[h,w,cout,b]
            hi = (h - 1) * sH + kh - pH # pozycja w "x" tak jak w forward (z uwzględnieniem kroku i marginesu)
            wi = (w - 1) * sW + kw - pW # pozycja w "W" tak jak w forward (z uwzględnieniem kroku i marginesu)
            if 1 <= hi <= H && 1 <= wi <= Wd # tylko dla rzeczywistych (niemarginalnych) pozycji
                # y = Σ W*x  ->  dL/dW = x,  dL/dx = W
                # Gradient to "go" (= dL/dy[h,w,cout,b]) pomnożony odpowiednio:
                gW[kH + 1 - kh, kW + 1 - kw, cin, cout] += go * x[hi, wi, cin, b] # dL/dW = Σ x*go po wszystkich pozycjach wejścia
                gx[hi, wi, cin, b] += go * W[kH + 1 - kh, kW + 1 - kw, cin, cout] # dL/dx = Σ W*go po wszystkich elementach okien które trafiły w tę pozycję
            end
        end
    end
    return (gx, gW)
end

# Splot zoptymalizowany (P0): schemat im2col + GEMM.
# Idea: splot zamieniany jest na jedno duże mnożenie macierzowe (BLAS):
#   1. "im2col" rozkłada otoczenia (receptive fields) wejścia do macierzy "cols",
#      gdzie kolumna = jedno okno splotu (jedna pozycja wyjścia),
#   2. jądro po flipie jest spłaszczane do macierzy "(K, Cout)", K = kH*kW*Cin,
#   3. wynik = "colsᵀ * Wm" liczony przez "mul!" (BLAS) zamiast pętli skalarnych.
# Konwencje (flip jądra, padding zerowy, stride) identyczne jak w wariancie naive.

# Linearyzacja indeksów (zgodna z układem column-major):
#   wiersz "cols":   row = kh + (kw-1)*kH + (cin-1)*kH*kW          (kh najszybciej zmienne)
#   kolumna "cols":  col = h + (w-1)*H_out + (b-1)*H_out*W_out     (h najszybciej zmienne)

# Wypełnia macierz "cols (K, N)" oknami splotu; pozycje poza wejściem dostają 0 (padding).
function _im2col!(cols::AbstractMatrix, x::AbstractArray{T,4}, kH, kW, sH, sW, pH, pW, H_out, W_out) where {T}
    H, Wd, Cin, B = size(x) # rozmiary wejścia (przestrzenne, kanały, batch)
    @inbounds for b = 1:B, w = 1:W_out, h = 1:H_out # jedna kolumna "cols" = jedno okno (h, w, b)
        col = h + (w - 1) * H_out + (b - 1) * H_out * W_out # indeks kolumny ("h" najszybciej zmienne)
        row = 0 # licznik wiersza; inkrementacja w kolejności (kh, kw, cin) - zgodnie z linearyzacją jądra
        for cin = 1:Cin, kw = 1:kW, kh = 1:kH # "kh" w pętli wewnętrznej = zapis po kolejnych wierszach kolumny
            row += 1
            hi = (h - 1) * sH + kh - pH # pozycja w "x" w osi "height" (krok i margines jak w naive)
            wi = (w - 1) * sW + kw - pW # pozycja w "x" w osi "width"
            cols[row, col] = (1 <= hi <= H && 1 <= wi <= Wd) ? x[hi, wi, cin, b] : zero(T) # zero poza brzegiem = padding
        end
    end
    return cols
end

# Rozrzuca (scatter-add) gradient okien "gcols (K, N)" z powrotem do kształtu wejścia "gx".
# Operacja odwrotna do "_im2col!"; "+=" bo jedna pozycja wejścia należy do wielu okien.
function _col2im!(gx::AbstractArray{T,4}, gcols::AbstractMatrix, kH, kW, sH, sW, pH, pW, H_out, W_out) where {T}
    H, Wd, Cin, B = size(gx) # rozmiary wejścia (cel rozrzucania)
    fill!(gx, zero(T)) # start od zer - wkłady są akumulowane
    @inbounds for b = 1:B, w = 1:W_out, h = 1:H_out # ta sama kolejność iteracji co w "_im2col!"
        col = h + (w - 1) * H_out + (b - 1) * H_out * W_out
        row = 0
        for cin = 1:Cin, kw = 1:kW, kh = 1:kH
            row += 1
            hi = (h - 1) * sH + kh - pH
            wi = (w - 1) * sW + kw - pW
            if 1 <= hi <= H && 1 <= wi <= Wd # pozycje paddingu nie wnoszą wkładu
                gx[hi, wi, cin, b] += gcols[row, col] # suma po wszystkich oknach zawierających (hi, wi)
            end
        end
    end
    return gx
end

# Spłaszcza jądro z flipem do macierzy "(K, Cout)": Wm[row, cout] = W[kH+1-kh, kW+1-kw, cin, cout].
# Flip zachowuje konwencję matematyczną splotu z wariantu naive (zgodną z Flux.Conv).
function _kernel2mat!(Wm::AbstractMatrix, W::AbstractArray{Tw,4}) where {Tw}
    kH, kW, Cin, Cout = size(W)
    @inbounds for cout = 1:Cout, cin = 1:Cin, kw = 1:kW, kh = 1:kH
        Wm[kh + (kw - 1) * kH + (cin - 1) * kH * kW, cout] = W[kH + 1 - kh, kW + 1 - kw, cin, cout] # flip indeksów (kh, kw)
    end
    return Wm
end

# Forward: y = colsᵀ * Wm przez BLAS, potem powrót do układu "(H_out, W_out, C_out, B)".
# Optymalizacja P1: bufory pośrednie (":cols", ":Wm", ":Y") pochodzą z przestrzeni
# roboczej warstwy "c.ws" i są reużywane między iteracjami; wynik pisany do "out"
# (bufor węzła grafu) albo do świeżej tablicy, gdy "out === nothing" / kształt się zmienił.
function _conv_forward!(out, c, x::AbstractArray{Tx,4}, W::AbstractArray{Tw,4}) where {Tx, Tw}
    pH, pW = c.pad # margines per wymiar
    sH, sW = c.stride # krok per wymiar
    kH, kW, Cin, Cout = size(W) # rozmiary jądra i kanały
    H, Wd, Cin2, B = size(x)
    Cin == Cin2 || throw(DimensionMismatch("Conv: Cin z wag != Cin z wejścia")) # zgodność liczby kanałów
    H_out = div(H  + 2pH - kH, sH) + 1 # rozmiar wyjścia w osi Height (formuła jak w naive)
    W_out = div(Wd + 2pW - kW, sW) + 1 # rozmiar wyjścia w osi Width
    T = promote_type(Tx, Tw) # wspólny typ obliczeń (np. Float32 x Float32 = Float32)
    K = kH * kW * Cin # liczba elementów jednego okna splotu
    N = H_out * W_out * B # liczba okien = liczba pozycji wyjścia
    cols = _fit!(c.ws, :cols, T, (K, N)) # bufor okien (reużywany)
    _im2col!(cols, x, kH, kW, sH, sW, pH, pW, H_out, W_out) # rozkład wejścia na okna
    Wm = _fit!(c.ws, :Wm, Tw, (K, Cout)) # bufor macierzy jądra (reużywany)
    _kernel2mat!(Wm, W) # jądro jako macierz "(K, Cout)" z flipem
    Y = _fit!(c.ws, :Y, T, (N, Cout)) # wynik GEMM: wiersz = pozycja wyjścia (h, w, b), kolumna = kanał
    mul!(Y, transpose(cols), Wm) # jedno mnożenie macierzowe (BLAS) zamiast 7 pętli
    o = _ensure(out, T, (H_out, W_out, Cout, B)) # bufor wyjścia węzła
    # "reshape (H_out, W_out, B, Cout)" odtwarza osie z linearyzacji kolumn; "permutedims!" przestawia w miejscu na "(H_out, W_out, C_out, B)"
    permutedims!(o, reshape(Y, H_out, W_out, B, Cout), (1, 2, 4, 3))
    return o
end
_conv_forward(c, x, W) = _conv_forward!(nothing, c, x, W) # wariant alokujący (pierwsze wywołanie / użycie poza grafem)

# Backward: dL/dW i dL/dx jako dwa GEMM-y na tych samych macierzach "cols"/"Wm".
#   gWm = cols * G   (K, Cout)  ->  unflip do kształtu jądra,
#   gcols = Wm * Gᵀ  (K, N)     ->  "_col2im!" rozrzuca do "gx".
# Optymalizacja P1: wszystkie tablice pośrednie oraz wyniki ("gx", "gW") pochodzą
# z przestrzeni roboczej "c.ws". Zwracane "gx"/"gW" są nadpisywane przy KOLEJNYM
# wywołaniu backward tej warstwy - są czytane tylko w obrębie jednego przejścia
# backward! (akumulacja do Variable kopiuje wartości), więc reużycie jest bezpieczne.
function _conv_backward(c, x::AbstractArray{Tx,4}, W::AbstractArray{Tw,4}, g::AbstractArray{Tg,4}) where {Tx, Tw, Tg}
    pH, pW = c.pad # margines
    sH, sW = c.stride # krok
    kH, kW, Cin, Cout = size(W) # rozmiary jądra
    H, Wd, _, B = size(x)
    H_out, W_out, _, _ = size(g) # rozmiary gradientu wyjścia (kształt y z forwarda)
    T = promote_type(Tx, Tw, Tg) # wspólny typ obliczeń pośrednich
    K = kH * kW * Cin
    N = H_out * W_out * B
    cols = _fit!(c.ws, :cols, T, (K, N)) # te same okna co w forward (liczone ponownie - graf nie cache'uje)
    _im2col!(cols, x, kH, kW, sH, sW, pH, pW, H_out, W_out)
    G = _fit!(c.ws, :G, T, (N, Cout)) # gradient wyjścia w układzie "(pozycja, kanał)" - spójnym z "Y" z forwarda
    permutedims!(reshape(G, H_out, W_out, B, Cout), g, (1, 2, 4, 3)) # "(H_out, W_out, C_out, B)" -> "(H_out, W_out, B, C_out)"
    # dL/dWm: każda kolumna "cols" wniosła wkład do wiersza "G" -> suma iloczynów po oknach
    gWm = _fit!(c.ws, :gWm, T, (K, Cout))
    mul!(gWm, cols, G) # GEMM: (K, N) * (N, Cout)
    gW = _fit!(c.ws, :gW, Tw, size(W)) # gradient jądra w oryginalnym kształcie (bufor reużywany)
    @inbounds for cout = 1:Cout, cin = 1:Cin, kw = 1:kW, kh = 1:kH
        gW[kH + 1 - kh, kW + 1 - kw, cin, cout] = gWm[kh + (kw - 1) * kH + (cin - 1) * kH * kW, cout] # unflip - odwrócenie "_kernel2mat!"
    end
    # dL/dcols: gradient każdego elementu okna = suma po kanałach wyjścia
    Wm = _fit!(c.ws, :Wm, Tw, (K, Cout))
    _kernel2mat!(Wm, W)
    gcols = _fit!(c.ws, :gcols, T, (K, N))
    mul!(gcols, Wm, transpose(G)) # GEMM: (K, Cout) * (Cout, N)
    gx = _fit!(c.ws, :gx, Tx, size(x)) # gradient wejścia w oryginalnym kształcie (bufor reużywany)
    _col2im!(gx, gcols, kH, kW, sH, sW, pH, pW, H_out, W_out) # scatter-add okien do pozycji wejścia
    return (gx, gW)
end

# Pomocnicze:
# bias ma kształt "(C_out, 1)", a potrzeba dodać go do tensora
# reshape "(H_out, W_out, C_out, B)" na "(1, 1, C_out, 1)" rozwiązuje ten problem przez rzutowanie.
_bias4d(b::AbstractArray) = reshape(b, 1, 1, length(b), 1)

# Wariant BEZ biasu: 
# 3 wejścia (config, x, W)
# Config "c" to "Constant(c::Conv)" mający "padding" i "stride" (nietrenowalne).
forward(::BroadcastedOperator{typeof(conv_op)}, c, x, W) = _conv_forward(c, x, W)
forward!(out, ::BroadcastedOperator{typeof(conv_op)}, c, x, W) = _conv_forward!(out, c, x, W) # wariant in-place (P1)
function backward(::BroadcastedOperator{typeof(conv_op)}, c, x, W, g)
    gx, gW = _conv_backward(c, x, W, g)
    return (nothing, gx, gW)   # krotka po KOLEJNOŚCI wejść: (g_c, g_x, g_W); "c" jest Constant -> nic nie zwraca
end

# Wariant Z biasem:
# 4 wejścia (config, x, W, b)
forward(::BroadcastedOperator{typeof(conv_op)}, c, x, W, b) =
    _conv_forward(c, x, W) .+ _bias4d(b) # bias rzutuje się po "(H_out, W_out, B)" (jedyny działający wymiar to C_out)
function forward!(out, ::BroadcastedOperator{typeof(conv_op)}, c, x, W, b) # wariant in-place (P1)
    o = _conv_forward!(out, c, x, W)
    o .+= _bias4d(b) # bias dodawany w miejscu
    return o
end
function backward(::BroadcastedOperator{typeof(conv_op)}, c, x, W, b, g)
    gx, gW = _conv_backward(c, x, W, g)                                      # gradienty po x i W liczymy normalnie (bias nie wpływa)
    # Bias dokłada się element-wise w wymiarze C_out -> dL/db[cout] to suma
    # gradientu wyjścia po wszystkich (h, w, batch) dla danego kanału "cout".
    Cout = size(g, 3) # liczba kanałów wyjścia
    gb = reshape([sum(@view g[:, :, cout, :]) for cout in 1:Cout], size(b)) # sumuje "g" po (H_out, W_out, B) dla każdego "cout"; reshape dopasowuje do oryginalnego size(b)
    #@view powoduje, że sumuje gradient po (H_out, W_out, B) bez kopiowania całego bloku g[:, :, cout, :] (oszczędza pamięć)
    return (nothing, gx, gW, gb) # (g_config, g_x, g_Weights, g_bias)
end

# MaxPool 2D
# Wymiary:
# wejście x : (H, W, C, B)  ->  wyjście y : (H_out, W_out, C, B)
# okno "pool = (kH, kW)",
# krok "stride", padding "pad" - wszystko per wymiar.
# Padding traktuje jak "-Inf" (wirtualnie), więc poza brzegiem nic nie może "wygrać" maxa - nie psuje to forwarda ani backwarda.

# Funkcja-znacznik, jak "conv_op" - istnieje tylko jako wartość do wyboru metody w "forward" / "backward".
function maxpool_op end

# Optymalizacja P1: wynik pisany do bufora "out" (węzła grafu); każda komórka "y"
# jest nadpisywana, więc bufor nie wymaga zerowania.
function _maxpool_forward!(out, m, x::AbstractArray{T,4}) where {T}
    kH, kW = m.pool # rozmiar okna
    sH, sW = m.stride # krok per wymiar
    pH, pW = m.pad # padding per wymiar
    H, Wd, C, B = size(x) # rozmiary wejścia (przestrzenne, kanały, batch)
    H_out = div(H + 2pH - kH, sH) + 1 # rozmiar wyjścia - identyczna formuła jak przy Conv
    W_out = div(Wd + 2pW - kW, sW) + 1
    y = _ensure(out, T, (H_out, W_out, C, B)) # bufor wyniku (reużywany między iteracjami)
    @inbounds for b = 1:B, c = 1:C, h = 1:H_out, w = 1:W_out # pętla po każdej pozycji wyjścia
        best = typemin(T) # lokalne maksimum dla bieżącego okna; "typemin(T)" - każda realna wartość wygrywa max
        for kh = 1:kH, kw = 1:kW # skan po oknie
            hi = (h - 1) * sH + kh - pH # pozycja w wejściu w osi "height"
            wi = (w - 1) * sW + kw - pW # pozycja w wejściu w osi "width"
            if 1 <= hi <= H && 1 <= wi <= Wd # pozycje poza brzegiem są ignorowane (są "-Inf")
                v = x[hi, wi, c, b]
                v > best && (best = v) # aktualizacja max w oknie
            end
        end
        y[h, w, c, b] = best # zapis maksimum dla tej komórki wyjścia
    end
    return y
end
_maxpool_forward(m, x) = _maxpool_forward!(nothing, m, x) # wariant alokujący (pierwsze wywołanie / użycie poza grafem)

# Backward dla MaxPool: 
#gradient wpada TYLKO do pozycji, która wygrała max
# (reguła łańcuchowa dla max: pochodna po zwycięskim argumencie = 1, reszta = 0).
# W przypadku remisu trafia do pierwszej znalezionej pozycji (drobna niestabilność gradientu na patologicznych wejściach)

function _maxpool_backward(m, x::AbstractArray{T,4}, g::AbstractArray{Tg,4}) where {T, Tg}
    kH, kW = m.pool # rozmiar okna per wymiar
    sH, sW = m.stride # krok per wymiar
    pH, pW = m.pad # padding per wymiar
    H, Wd, C, B = size(x) # rozmiary wejścia (przestrzenne, kanały, batch)
    H_out, W_out, _, _ = size(g) # rozmiary gradientu wyjścia (odpowiada kształtowi y z forwarda)
    # Optymalizacja P1: bufor ":gx" z przestrzeni roboczej warstwy (reużywany między
    # iteracjami); czytany tylko w obrębie jednego przejścia backward!, więc reużycie jest bezpieczne.
    gx = _fit!(m.ws, :gx, T, size(x))
    fill!(gx, zero(T)) # start od zer; wkłady tylko do pozycji argmax (wygrywających max)
    # argmax jest liczony "w locie" (drugi raz) zamiast pamiętać go z forwarda.
    # To oszczędza pamięć; narzut CPU przy standardowych oknach (2 x 2) jest pomijalny.
    @inbounds for b = 1:B, c = 1:C, h = 1:H_out, w = 1:W_out # pętla po wszystkich pozycjach wyjścia
        best = typemin(T) # aktualny max w oknie
        bhi, bwi = 0, 0 # pozycja aktualnego maxa (0 = jeszcze nie znaleziono nic realnego)
        for kh = 1:kH, kw = 1:kW
            hi = (h - 1) * sH + kh - pH # pozycja w wejściu w osi "height"
            wi = (w - 1) * sW + kw - pW # pozycja w wejściu w osi "width"
            if 1 <= hi <= H && 1 <= wi <= Wd
                v = x[hi, wi, c, b]
                if v > best
                    best = v; bhi = hi; bwi = wi # aktualizacja max w oknie
                end
            end
        end
        # przekazanie całego gradientu tylko do zwycięzcy okna (reguła łańcuchowa):
        bhi > 0 && (gx[bhi, bwi, c, b] += g[h, w, c, b]) # "+=" bo ta pozycja mogła wygrać w wielu oknach
    end
    return (gx,) # krotka jednoelementowa (dla spójności API z innymi operatorami)
end

# 2 wejścia operatora: "Constant(m::MaxPool)" (niesie pool/stride/pad) i "x".
forward(::BroadcastedOperator{typeof(maxpool_op)}, m, x) = _maxpool_forward(m, x)
forward!(out, ::BroadcastedOperator{typeof(maxpool_op)}, m, x) = _maxpool_forward!(out, m, x) # wariant in-place (P1)
function backward(::BroadcastedOperator{typeof(maxpool_op)}, m, x, g)
    gx, = _maxpool_backward(m, x, g) # rozpakowanie krotki 1-el.
    return (nothing, gx) # po kolejności: (g_m, g_x); "m" jest Constantem -> nothing
end

# Flatten
# Wymiary:
# wejście x : (D1, D2, …, D_{n-1}, B)  ->  wyjście y : (D1*…*D_{n-1}, B)
# Operacja CZYSTO typologiczna: "reshape" bez kopiowania wartości.
# Backward to ten sam reshape w drugą stronę - gradient "nie wycieka"
# (wszystkie elementy pozostają na swoich miejscach w pamięci)
function flatten_op end

_flatten_forward(x::AbstractArray) = reshape(x, :, size(x, ndims(x))) # ":" = "autodomyślny rozmiar" (iloczyn wszystkich pozostałych wymiarów); zachowuje ostatni wymiar (batch) jako kolumny
_flatten_backward(x::AbstractArray, g::AbstractArray) = (reshape(g, size(x)),)  # cofnięcie reshape do kształtu "x"

forward(::BroadcastedOperator{typeof(flatten_op)}, x) = _flatten_forward(x)
function backward(::BroadcastedOperator{typeof(flatten_op)}, x, g)
    gx, = _flatten_backward(x, g)   # rozpakowanie krotki 1-el.
    return (gx,)
end

# Dropout
# Wariant "inverted dropout":
# przy treningu każdy element jest zerowany z prawdopodobieństwem "p", a pozostałe skalowane przez "1/(1-p)", tak żeby WARTOŚĆ OCZEKIWANA pozostała bez zmian.
# Dzięki temu w trybie ewaluacji nic nie trzeba skalować - dropout = identyczność.

# Maskę zapisuje w "layer.mask", bo "backward" musi użyć DOKŁADNIE tej samej maski co "forward"
# (losowanie raz na forward -> zapamiętanie -> użycie w backward).
function dropout_op end

function forward(::BroadcastedOperator{typeof(dropout_op)}, layer, x)
    if !layer.active # tryb ewaluacji (testmode!)
        layer.mask = nothing # wyzerowanie maski - w backward jako "pass-through"
        return copy(x) # kopia żeby wyjście było fizycznie osobną tablicą (dla bezpieczeństwa)
    end
    T = eltype(x) # typ elementu (np. Float32)
    p = T(layer.p) # konwersja "p::Float64" -> "T"
    mask = rand(T, size(x)) .> p # losowanie liczb losowanych z rozkładu jednostajnego [0,1), porównanie ".> p" daje "BitArray" kształtu x (true gdy zachowuje się element, false gdy jest zerowany)
    layer.mask = mask # CACHE dla backwardu - ta sama maska
    scale = one(T) / (one(T) - p) # "inverted" to skalowanie 1/(1-p) żeby wartość oczekiwana (średnia teoretyczna) E[y] = E[x]
    return x .* mask .* scale # Kazdy element, który mask=true, jest zachowany, resztę zeruje
end

# Dropout w miejscu (P1): losowanie do bufora ":rand" (rand! bez alokacji),
# maska reużywana między iteracjami, wynik pisany do bufora "out".
function forward!(out, ::BroadcastedOperator{typeof(dropout_op)}, layer, x)
    T = eltype(x)
    o = _ensure(out, T, size(x)) # bufor wyjścia węzła
    if !layer.active # tryb ewaluacji - identyczność
        layer.mask = nothing # sygnał "pass-through" dla backward
        o .= x # kopia w miejscu (wyjście pozostaje osobną tablicą)
        return o
    end
    p = T(layer.p)
    r = _fit!(layer.ws, :rand, T, size(x)) # bufor liczb losowych (reużywany)
    rand!(r) # losowanie z [0,1) w miejscu
    mask = layer.mask
    if !(mask isa BitArray) || size(mask) != size(x) # maska reużywana, realokacja tylko przy zmianie kształtu
        mask = BitArray(undef, size(x))
        layer.mask = mask
    end
    mask .= r .> p # aktualizacja maski w miejscu (true = element zachowany)
    scale = one(T) / (one(T) - p) # skalowanie "inverted dropout" - E[y] = E[x]
    o .= x .* mask .* scale
    return o
end

function backward(::BroadcastedOperator{typeof(dropout_op)}, layer, x, g)
    if !layer.active || layer.mask === nothing # eval-mode albo forward jeszcze się nie odbył -> gradient przechodzi bez zmian (pass-through)
        return (nothing, g) # (g_layer, g_x); "layer" to Constant -> nothing
    end
    T = eltype(g) # typ gradientu (dopasowany do batcha)
    scale = one(T) / (one(T) - T(layer.p)) # to samo skalowanie co w forward (symetria)
    return (nothing, g .* layer.mask .* scale) # dy/dx = mask*scale -> gradient też tylko na aktywnych pozycjach i przeskalowany
end
