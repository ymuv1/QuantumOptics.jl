names = readdir(".")

for name=names
    if beginswith(name, "test_") && endswith(name, ".jl")
        println("Run $name")
        include(name)
    end
end
