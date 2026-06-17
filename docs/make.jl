using TomoBOS
using Documenter

DocMeta.setdocmeta!(TomoBOS, :DocTestSetup, :(using TomoBOS); recursive=true)

makedocs(;
    modules=[TomoBOS],
    authors="Akamine",
    sitename="TomoBOS.jl",
    format=Documenter.HTML(;
        canonical="https://wavepackets.github.io/TomoBOS.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/wavepackets/TomoBOS.jl",
    devbranch="main",
)
