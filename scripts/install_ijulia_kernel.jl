# Rejestruje kernel Jupyter z `julia --project=<katalog AWID>`.
using Pkg
const ROOT = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)
using IJulia
kernel = "Julia (AWID)"
installkernel(kernel, "--project=$ROOT")
println("Zainstalowano kernel: ", kernel, " z --project=", ROOT)
