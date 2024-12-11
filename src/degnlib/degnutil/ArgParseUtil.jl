#!/usr/bin/env julia
module ArgParseUtil

using ArgParse
using NamedTupleTools

export parse_arguments

get_field_groups(settings) = [field.group for field in settings.args_table.fields]
get_field_dest_names(settings) = [field.dest_name for field in settings.args_table.fields]

"Returns (positionals, optionals) as named tuples or if settings is for command with a subcommand we return (command_name::String, (positionals, optionals))."
function parse_arguments(settings)
    parsed_args = (;parse_args(settings; as_symbols=true)...) # syntax to convert dict to named tuple
    # has subcommands?
    if keys(parsed_args)[1] == :_COMMAND_
        cmd = parsed_args[:_COMMAND_]
        parsed_args = (;parsed_args[cmd]...)  # syntax to convert dict to named tuple
        settings = settings.args_table.subsettings[String(cmd)]  # get subcommand settings
        return String(cmd), split_positionals(settings, parsed_args)
    else
        return split_positionals(settings, parsed_args)
    end
end

"Split positionals from optionals so we return a tuple where the first element is positionals and the second element is named tuple of optionals."
function split_positionals(settings, parsed_args)
    positional_keys = get_field_dest_names(settings)[get_field_groups(settings) .== "positional"]
    split(parsed_args, Tuple(Symbol.(positional_keys)))
end

function main(settings, f)
    positionals, optionals = parse_arguments(settings)
    @info("Positionals: $positionals")
    @info("Optionals: $optionals")
    f(positionals...; optionals...)
end


end;
