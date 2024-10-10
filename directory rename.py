def conv(string):
    return string.replace('\\', '/')

u_in = str(input(r"Directory: "))

print(conv(u_in))