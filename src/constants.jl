q_aa = 21

let pos::Int = 1
    global current_pos() = pos
    global increment_pos() = (pos += 1)
    global set_pos(i::Int) = (pos = i)
    global reset_pos() = set_pos(0)
end

let v::Int = 0
    global set_verbose(val) = (v=val)
    global verbose() = v
end

