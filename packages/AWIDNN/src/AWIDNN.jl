# src/AWIDNN.jl
module AWIDNN

include("structures.jl")
include("autodiff.jl")
include("layers.jl")

# typy grafu
export GraphNode, Operator, Constant, Variable, ScalarOperator, BroadcastedOperator

# Automatyczne różniczkowanie (AD) wsteczne
export graph, zerograd!, forward!, backward!, optimize!
export conv_op, maxpool_op, flatten_op, dropout_op

# warstwy i model
export Chain, Dense, Embedding, Conv, MaxPool, FlattenLayer, flatten, Dropout
export trainmode!, testmode!

# funkcje straty
export bce, bce_el, sum_node, logitcrossentropy

# funkcje aktywacji
export relu, σ

# pomocnicze (zamienniki Flux)
export DataLoader, onehotbatch, onecold
export Descent, setup, update!

end # module
