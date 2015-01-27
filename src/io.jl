module io

function guess_type(x)
    try
        return eval(parse(x))
    catch
        return x
    end
end

function split(word::AbstractString, delimiter::AbstractString)
    words = []
    lastsearchpos = -1:0
    while true
        searchpos = search(word, delimiter, lastsearchpos[end]+1)
        if searchpos==0:-1
            push!(words, word[lastsearchpos[end]+1:end])
            break
        end
        push!(words, word[lastsearchpos[end]+1:searchpos[1]-1])
        lastsearchpos = searchpos
    end
    return words
end

function join(delimiter::AbstractString, words::Vector)
    result = words[1]
    for i=2:length(words)
        result = result*delimiter*words[i]
    end
    return result
end

function dict2filename(d::Dict; extension=".dat")
    pairs = ["$(key)=$(item)" for (key, item) in d]
    return join(";", sort(pairs))*extension
end

function filename2dict(name::AbstractString; extension=".dat")
    @assert endswith(name, extension)
    items = split(name[1:end-length(extension)], ";")
    D = Dict()
    for item in items
        array = split(item, "=")
        key = array[1]
        value = array[2]
        D[key] = guess_type(value)
    end
    return D
end

end # module
