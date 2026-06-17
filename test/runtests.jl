using TomoBOS
using Test
using Aqua
using JET

@testset "TomoBOS.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(TomoBOS)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(TomoBOS; target_defined_modules = true)
    end
    # Write your tests here.
end
