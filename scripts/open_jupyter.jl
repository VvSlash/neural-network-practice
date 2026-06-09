# Uruchamia Jupyter (notatnik / lab) z aktywnym środowiskiem katalogu AWID3.
using Pkg
const ROOT = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)
using IJulia
notebook(dir=ROOT)
