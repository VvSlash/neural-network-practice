# scripts/benchmark_bottlenecks.jl
# Benchmarki wąskich gardeł B1–B9 z "PORÓWNANIE-IMPLEMENTACJI.md" (sekcja 6.3).
# Uruchomienie z katalogu głównego repozytorium: "julia scripts/benchmark_bottlenecks.jl"
#
# Skrypt działa zarówno PRZED jak i PO optymalizacji:
# - warianty referencyjne "_naive" są benchmarkowane tylko, gdy istnieją w bibliotece
#   (po optymalizacji P0 służą do porównania "przed/po" oraz testu poprawności),
# - dane są syntetyczne (bez MLDatasets) o rozmiarach identycznych jak w sieci
#   z notatnika "AWID-2026-CNN-AWIDNN.ipynb" (batchsize = 10).

using Pkg
Pkg.activate(joinpath(@__DIR__, "..")) # środowisko główne repozytorium (BenchmarkTools + AWIDNN dev)

using BenchmarkTools
using Profile
using Random
using AWIDNN

Random.seed!(1) # powtarzalność danych syntetycznych między uruchomieniami

println("Julia $VERSION | watki: $(Threads.nthreads())")
println("="^78)

# Rozmiary tensorów jak w sieci z notatnika (batchsize = 10)
const BATCH = 10
x1 = randn(Float32, 28, 28, 1, BATCH)  # wejście conv1: (H, W, C_in, B)
x2 = randn(Float32, 14, 14, 6, BATCH)  # wejście conv2 (po maxpool1)
c1 = Conv((3, 3), 1 => 6,  pad = 1, bias = false) # konfiguracja conv1 (pad/stride/wagi)
c2 = Conv((3, 3), 6 => 16, pad = 1, bias = false) # konfiguracja conv2
g1 = randn(Float32, 28, 28, 6, BATCH)  # gradient "z góry" dla wyjścia conv1
g2 = randn(Float32, 14, 14, 16, BATCH) # gradient "z góry" dla wyjścia conv2

# Flaga: czy biblioteka zawiera zachowane implementacje referencyjne (po P0)
const HAS_NAIVE = isdefined(AWIDNN, :_conv_forward_naive)

# B1 - splot (dominujące wąskie gardło)

println("\n[B1] Splot: _conv_forward / _conv_backward")
print("  conv1 forward  (28x28x1x10  * 3x3x1x6):   ")
@btime AWIDNN._conv_forward($c1, $x1, $(c1.weight))
print("  conv2 forward  (14x14x6x10  * 3x3x6x16):  ")
@btime AWIDNN._conv_forward($c2, $x2, $(c2.weight))
print("  conv1 backward:                            ")
@btime AWIDNN._conv_backward($c1, $x1, $(c1.weight), $g1)
print("  conv2 backward:                            ")
@btime AWIDNN._conv_backward($c2, $x2, $(c2.weight), $g2)

if HAS_NAIVE
    println("  --- wariant referencyjny (naiwne pętle, przed P0) ---")
    print("  conv1 forward  (naive):                    ")
    @btime AWIDNN._conv_forward_naive($c1, $x1, $(c1.weight))
    print("  conv2 forward  (naive):                    ")
    @btime AWIDNN._conv_forward_naive($c2, $x2, $(c2.weight))
    print("  conv1 backward (naive):                    ")
    @btime AWIDNN._conv_backward_naive($c1, $x1, $(c1.weight), $g1)
    print("  conv2 backward (naive):                    ")
    @btime AWIDNN._conv_backward_naive($c2, $x2, $(c2.weight), $g2)
end

# Poprawność - splot zoptymalizowany vs referencyjny + gradient numeryczny

if HAS_NAIVE
    println("\n[Poprawnosc] im2col+GEMM vs naiwne petle")
    # zgodność forward na rozmiarach z sieci
    y_opt = AWIDNN._conv_forward(c2, x2, c2.weight)
    y_ref = AWIDNN._conv_forward_naive(c2, x2, c2.weight)
    println("  forward  max|roznica| = ", maximum(abs.(y_opt .- y_ref)))
    @assert isapprox(y_opt, y_ref; rtol = 1f-5) "forward: rozjazd wynikow"
    # zgodność backward (gx i gW)
    gx_o, gW_o = AWIDNN._conv_backward(c2, x2, c2.weight, g2)
    gx_r, gW_r = AWIDNN._conv_backward_naive(c2, x2, c2.weight, g2)
    println("  backward max|roznica| gx = ", maximum(abs.(gx_o .- gx_r)),
            ", gW = ", maximum(abs.(gW_o .- gW_r)))
    @assert isapprox(gx_o, gx_r; rtol = 1f-4) "backward gx: rozjazd"
    @assert isapprox(gW_o, gW_r; rtol = 1f-4) "backward gW: rozjazd"
    # gradient numeryczny (centralne różnice) na małym tensorze: L = sum(y .* r)
    let xs = randn(Float64, 5, 5, 2, 2), Ws = randn(Float64, 3, 3, 2, 3),
        cs = Conv((3, 3), 2 => 3, pad = 1, bias = false, T = Float64)
        r = randn(Float64, size(AWIDNN._conv_forward(cs, xs, Ws)))
        gx, gW = AWIDNN._conv_backward(cs, xs, Ws, r) # gradient analityczny dla seedu "r"
        eps = 1e-6
        # dL/dx w losowym indeksie
        idx = CartesianIndex(3, 4, 1, 2)
        xp = copy(xs); xp[idx] += eps; xm = copy(xs); xm[idx] -= eps
        num = (sum(AWIDNN._conv_forward(cs, xp, Ws) .* r) - sum(AWIDNN._conv_forward(cs, xm, Ws) .* r)) / 2eps
        println("  grad. numeryczny dL/dx: analityczny = ", gx[idx], ", numeryczny = ", num)
        @assert isapprox(gx[idx], num; rtol = 1e-6) "gradient numeryczny dx: rozjazd"
        # dL/dW w losowym indeksie
        idxw = CartesianIndex(2, 1, 2, 3)
        Wp = copy(Ws); Wp[idxw] += eps; Wm = copy(Ws); Wm[idxw] -= eps
        numw = (sum(AWIDNN._conv_forward(cs, xs, Wp) .* r) - sum(AWIDNN._conv_forward(cs, xs, Wm) .* r)) / 2eps
        println("  grad. numeryczny dL/dW: analityczny = ", gW[idxw], ", numeryczny = ", numw)
        @assert isapprox(gW[idxw], numw; rtol = 1e-6) "gradient numeryczny dW: rozjazd"
    end
    println("  OK - wyniki zgodne")
end

# Poprawność pełnej ścieżki grafu (jądra in-place backward! + akumulacja, P1/B4):
# gradient numeryczny (centralne różnice) porównany z gradientami Variable po backward!.
# Sieć bez Dropoutu (losowa maska uniemożliwia różnice skończone) i w Float64.

println("\n[Poprawnosc] gradienty na pelnym grafie (sciezka in-place backward!)")
let
    Random.seed!(7)
    netv = Chain(
        Conv((3, 3), 1 => 2, pad = 1, bias = true, T = Float64), # bias=true - pokrywa wariant backward! z biasem
        MaxPool((2, 2)),
        flatten,
        Dense(32 => 5, relu; T = Float64),
        Dense(5 => 3; T = Float64),
    )
    xs = randn(Float64, 8, 8, 1, 4)
    ys = zeros(Float64, 3, 4); for j in 1:4; ys[rand(1:3), j] = 1.0; end
    inp = Constant(copy(xs)); tgt = Constant(ys)
    L = AWIDNN.logitcrossentropy(netv(inp), tgt)
    gg = graph(L)
    forward!(gg) # pierwszy przebieg alokuje bufory; kolejne idą ścieżką in-place
    forward!(gg); backward!(gg)
    eps = 1e-6
    maxrel = 0.0
    for v in gg # sprawdzenie losowego elementu każdego parametru (wagi + biasy)
        v isa Variable || continue
        length(v.output) == 0 && continue
        idx = rand(CartesianIndices(v.output))
        orig = v.output[idx]
        v.output[idx] = orig + eps; Lp = forward!(gg) # strata po perturbacji +eps
        v.output[idx] = orig - eps; Lm = forward!(gg) # strata po perturbacji -eps
        v.output[idx] = orig
        num = (Lp - Lm) / (2 * eps) # pochodna numeryczna (centralna)
        ana = v.gradient[idx] # pochodna analityczna z backward!
        rel = abs(num - ana) / max(abs(num), abs(ana), 1e-12)
        maxrel = max(maxrel, rel)
        println("  ", rpad(v.name, 8), " analityczny = ", round(ana, sigdigits = 8),
                ", numeryczny = ", round(num, sigdigits = 8))
    end
    @assert maxrel < 1e-5 "gradient grafu: rozjazd wzgledny $maxrel"
    println("  OK - max wzgledna roznica = ", round(maxrel, sigdigits = 3))
end

# B2-B5 - pełny krok treningowy na grafie (dispatch, alokacje forward/backward/zerograd)

println("\n[B2-B5] Pelny krok treningowy: forward! + backward! + optimize!")
net = Chain(
    Conv((3, 3), 1 => 6,  pad = 1, bias = false),
    MaxPool((2, 2)),
    Conv((3, 3), 6 => 16, pad = 1, bias = false),
    MaxPool((2, 2)),
    flatten,
    Dense(784 => 84, relu),
    Dropout(0.4),
    Dense(84 => 10),
)
input  = Constant(randn(Float32, 28, 28, 1, BATCH)) # bufor wejścia (jak w notatniku)
target = Constant(zeros(Float32, 10, BATCH))        # bufor etykiet one-hot
for j in 1:BATCH; target.output[rand(1:10), j] = 1f0; end
ŷ = net(input)                                      # budowa grafu (raz)
L = AWIDNN.logitcrossentropy(ŷ, target)             # korzeń grafu = strata
gr = graph(L)                                       # porządek topologiczny

print("  forward!:                ")
@btime forward!($gr)
print("  backward!:               ")
@btime backward!($gr)
print("  caly krok (fwd+bwd+SGD): ")
@btime (forward!($gr); backward!($gr); optimize!($gr, 1f-2))
# szacunek czasu epoki: 6000 batchy * czas kroku (bez DataLoadera i ewaluacji)
step_time = @belapsed (forward!($gr); backward!($gr); optimize!($gr, 1f-2))
println("  szacunkowy czas epoki (6000 batchy): ", round(step_time * 6000; digits = 1), " s")

# B6 - MaxPool (cache argmax z forward zamiast ponownego skanu w backward)

println("\n[B6] MaxPool: _maxpool_forward / _maxpool_backward")
m = MaxPool((2, 2))
xp = randn(Float32, 28, 28, 6, BATCH)
gp = randn(Float32, 14, 14, 6, BATCH)
AWIDNN._maxpool_forward(m, xp) # wypełnienie cache argmax (wymagane przed backward)
print("  forward  (28x28x6x10):   ")
@btime AWIDNN._maxpool_forward($m, $xp)
print("  backward z cache (B6):   ")
@btime AWIDNN._maxpool_backward($m, $xp, $gp)
if isdefined(AWIDNN, :_maxpool_backward_recompute_into!)
    gx_c = AWIDNN._fit!(m.ws, :gx, Float32, size(xp))
    gx_r = similar(xp)
    AWIDNN._maxpool_backward_into!(gx_c, m, gp)
    AWIDNN._maxpool_backward_recompute_into!(gx_r, m, xp, gp)
    println("  poprawnosc cache vs recompute: max|roznica| = ", maximum(abs.(gx_c .- gx_r)))
    @assert isapprox(gx_c, gx_r; rtol = 1f-6) "MaxPool backward: rozjazd cache vs recompute"
    print("  backward recompute (przed B6): ")
    @btime AWIDNN._maxpool_backward_recompute_into!($gx_r, $m, $xp, $gp)
end

# B8 - DataLoader (kopie batchy i kopia całego zbioru przy shuffle)

println("\n[B8] DataLoader: kopie danych")
Xfull = randn(Float32, 28, 28, 1, 60_000) # zbiór wielkości FashionMNIST (~188 MB)
Yfull = zeros(Float32, 10, 60_000)
print("  konstruktor shuffle=true (kopia calego zbioru): ")
@btime AWIDNN.DataLoader(($Xfull, $Yfull); batchsize = 10, shuffle = true) samples = 5 evals = 1
dl = AWIDNN.DataLoader((Xfull, Yfull); batchsize = 10, shuffle = false)
print("  pobranie jednego batcha (kopia wycinka):        ")
@btime first($dl)

# B9 - Dropout (alokacje maski i tymczasowej tablicy rand)

println("\n[B9] Dropout: forward (trening)")
d = Dropout(0.4)
xd = randn(Float32, 84, BATCH)
node_d = AWIDNN.BroadcastedOperator(AWIDNN.dropout_op, Constant(d), Constant(xd); name = "dropout")
print("  forward (84x10, p=0.4):  ")
@btime AWIDNN.forward($node_d, $d, $xd)
if isdefined(AWIDNN, :_fit!) # wariant in-place dostępny od optymalizacji P1
    out_d = AWIDNN.forward(node_d, d, xd) # bufor wyjścia dla jądra in-place
    print("  forward! in-place (P1):  ")
    @btime AWIDNN.forward!($out_d, $node_d, $d, $xd)
end

# Profil - rozkład czasu pełnego kroku treningowego (top wpisy z plików AWIDNN)
# Odpowiednik tekstowy @profview: te same dane Profile, posortowane po liczbie próbek.
# Interaktywny flamegraph: w VS Code / IJulia uruchomić "@profview for _ in 1:50 ... end".

println("\n[Profil] Top funkcje AWIDNN w pelnym kroku treningowym (50 krokow)")
Profile.clear()
@profile for _ in 1:50
    forward!(gr); backward!(gr); optimize!(gr, 1f-2)
end
buf = IOBuffer()
Profile.print(IOContext(buf, :displaysize => (1000, 240)); format = :flat, sortedby = :count)
profile_lines = split(String(take!(buf)), '\n')
awidnn_lines = filter(l -> occursin("AWIDNN", l) || occursin("autodiff.jl", l), profile_lines)
# format :flat sortuje rosnąco po liczbie próbek - największe wpisy są na końcu listy
for line in reverse(last(awidnn_lines, 14))
    println(rstrip(line))
end

# Makro @profview - interaktywny flamegraph wąskiego gardła B1 (zakomentowane po testach).
# Wymaga środowiska graficznego: VS Code (rozszerzenie Julia) albo ProfileView.jl.
# Aby ponownie zbadać wąskie gardło: odkomentować i uruchomić w REPL-u VS Code.
# @profview for _ in 1:200
#     AWIDNN._conv_forward(c2, x2, c2.weight)        # profil samego forwarda splotu
#     AWIDNN._conv_backward(c2, x2, c2.weight, g2)   # profil samego backwarda splotu
# end
# @profview for _ in 1:50
#     forward!(gr); backward!(gr); optimize!(gr, 1f-2) # profil pełnego kroku treningowego
# end

println("\n", "="^78)
println("Koniec benchmarkow.")
