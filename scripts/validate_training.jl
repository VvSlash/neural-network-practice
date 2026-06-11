# scripts/validate_training.jl
# Test regresji po optymalizacji: pełny trening jak w "AWID-2026-CNN-AWIDNN.ipynb"
# (3 epoki, batchsize = 10, SGD eta = 1e-2). Kryterium: ~86% dokładności na zbiorze
# testowym po 3 epokach oraz czas epoki znacząco krótszy niż przed optymalizacją.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MLDatasets, AWIDNN, Random
Random.seed!(42) # powtarzalność inicjalizacji wag i tasowania

train_data = MLDatasets.FashionMNIST(split = :train)
test_data  = MLDatasets.FashionMNIST(split = :test)

function loader(data; batchsize::Int = 1, shuffle::Bool = true)
    x4dim = reshape(Float32.(data.features), 28, 28, 1, :) # trywialny 4-wymiarowy kanał
    yhot = AWIDNN.onehotbatch(data.targets, 0:9) # 10 x N one-hot
    return AWIDNN.DataLoader((x4dim, yhot); batchsize, shuffle)
end

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

function loss_and_accuracy(model, data; batchsize::Int = 100)
    testmode!(model) # Dropout w trybie ewaluacji
    dl = loader(data; batchsize = batchsize, shuffle = false)
    total_loss, total_correct, total = 0f0, 0, 0
    for (x, y) in dl
        input = Constant(x)
        target = Constant(y)
        ŷ_node = model(input)
        L_node = AWIDNN.logitcrossentropy(ŷ_node, target)
        g = graph(L_node)
        forward!(g)
        bs = size(x)[end]
        total_loss += L_node.output * bs
        total_correct += sum(AWIDNN.onecold(ŷ_node.output) .== AWIDNN.onecold(target.output))
        total += bs
    end
    trainmode!(model, true)
    return (; loss = total_loss / total, acc = round(100 * total_correct / total; digits = 2))
end

settings = (; eta = 1f-2, epochs = 3, batchsize = 10)
input  = Constant(zeros(Float32, 28, 28, 1, settings.batchsize)) # bufor wejścia nadpisywany w pętli
target = Constant(zeros(Float32, 10, settings.batchsize)) # bufor etykiet nadpisywany w pętli
trainmode!(net, true)
ŷ_node = net(input) # graf budowany raz
L_node = AWIDNN.logitcrossentropy(ŷ_node, target)
g_train = graph(L_node)

println("Trening: 3 epoki, batchsize=10, eta=1e-2 (jak w notatniku)")
for epoch in 1:settings.epochs
    @time for (x, y) in loader(train_data; batchsize = settings.batchsize, shuffle = true)
        size(x)[end] == settings.batchsize || continue # pominięcie niepełnego batcha
        input.output  .= x
        target.output .= y
        forward!(g_train)
        backward!(g_train)
        optimize!(g_train, settings.eta)
    end
    tr = loss_and_accuracy(net, train_data)
    te = loss_and_accuracy(net, test_data)
    println("epoka $epoch: train acc = $(tr.acc)%, test acc = $(te.acc)%")
end
