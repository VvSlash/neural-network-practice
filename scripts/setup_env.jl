using Pkg
root = dirname(@__DIR__)
Pkg.activate(root)
Pkg.add(["IJulia", "MLDatasets", "Statistics", "CairoMakie", "Flux", "TensorOperations"])
Pkg.develop(path=joinpath(root, "packages", "AWIDNN"))
Pkg.precompile()
println("OK: środowisko w ", root)
